#include "obs-utils.h"
#include "plugin-support.h"

#include <obs-module.h>

#if defined(_WIN32)
#include <d3d11.h>
#include <cstring>
#endif

static gs_texture_t *ensure_effect_work_texture(filter_data *tf, uint32_t width, uint32_t height)
{
	if (tf->effectWorkTexture &&
	    (tf->effectWorkTextureWidth != width || tf->effectWorkTextureHeight != height)) {
		gs_texture_destroy(tf->effectWorkTexture);
		tf->effectWorkTexture = nullptr;
		tf->effectWorkTextureWidth = 0;
		tf->effectWorkTextureHeight = 0;
	}

	if (!tf->effectWorkTexture) {
		tf->effectWorkTexture = gs_texture_create(width, height, GS_BGRA, 1, nullptr, 0);
		if (!tf->effectWorkTexture) {
			return nullptr;
		}
		tf->effectWorkTextureWidth = width;
		tf->effectWorkTextureHeight = height;
	}

	return tf->effectWorkTexture;
}

#if defined(_WIN32)
static bool copyD3D11TextureToBGRA(gs_texture_t *texture, uint32_t width, uint32_t height,
				 cv::Mat &outBGRA)
{
	if (!texture || width == 0 || height == 0) {
		return false;
	}

	ID3D11Texture2D *texture2d = reinterpret_cast<ID3D11Texture2D *>(gs_texture_get_obj(texture));
	if (!texture2d) {
		return false;
	}

	ID3D11Device *device = nullptr;
	texture2d->GetDevice(&device);
	if (!device) {
		return false;
	}

	ID3D11DeviceContext *context = nullptr;
	device->GetImmediateContext(&context);
	if (!context) {
		device->Release();
		return false;
	}

	D3D11_TEXTURE2D_DESC staging_desc{};
	staging_desc.Width = width;
	staging_desc.Height = height;
	staging_desc.MipLevels = 1;
	staging_desc.ArraySize = 1;
	staging_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	staging_desc.SampleDesc.Count = 1;
	staging_desc.SampleDesc.Quality = 0;
	staging_desc.Usage = D3D11_USAGE_STAGING;
	staging_desc.BindFlags = 0;
	staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	staging_desc.MiscFlags = 0;

	ID3D11Texture2D *staging_texture = nullptr;
	HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
	if (FAILED(hr)) {
		context->Release();
		device->Release();
		return false;
	}

	context->CopyResource(staging_texture, texture2d);
	D3D11_MAPPED_SUBRESOURCE mapped{};
	hr = context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped);
	if (SUCCEEDED(hr)) {
		outBGRA.create(static_cast<int>(height), static_cast<int>(width), CV_8UC4);
		const uint32_t row_size = width * 4u;
		uint8_t *dst = outBGRA.data;
		const uint8_t *src = static_cast<const uint8_t *>(mapped.pData);
		for (uint32_t row = 0; row < height; ++row) {
			std::memcpy(dst + row * row_size, src + row * mapped.RowPitch, row_size);
		}
		context->Unmap(staging_texture, 0);
	}

	staging_texture->Release();
	context->Release();
	device->Release();
	return SUCCEEDED(hr);
}
#endif

bool getRGBAFromRenderTexture(filter_data *tf, uint32_t &width, uint32_t &height, bool readbackToCpu)
{
	if (!obs_source_enabled(tf->source)) {
		return false;
	}

	obs_source_t *target = obs_filter_get_target(tf->source);
	if (!target) {
		return false;
	}
	width = obs_source_get_base_width(target);
	height = obs_source_get_base_height(target);
	if (width == 0 || height == 0) {
		return false;
	}

	gs_texrender_reset(tf->texrender);
	if (!gs_texrender_begin(tf->texrender, width, height)) {
		return false;
	}
	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f,
		 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(tf->texrender);

	gs_texture_t *rendered_texture = gs_texrender_get_texture(tf->texrender);
	if (!rendered_texture) {
		return false;
	}

	if (!readbackToCpu) {
		return true;
	}

	cv::Mat frameBGRA;
#if defined(_WIN32)
	if (copyD3D11TextureToBGRA(rendered_texture, width, height, frameBGRA)) {
		std::lock_guard<std::mutex> lock(tf->inputBGRALock);
		tf->inputBGRA = std::move(frameBGRA);
		return true;
	}
#endif
	return false;
}

/**
  * @brief Get RGBA from the stage surface
  *
  * @param tf  The filter data
  * @param width  The width of the stage surface (output)
  * @param height  The height of the stage surface (output)
  * @return true  if successful
  * @return false if unsuccessful
*/
bool getRGBAFromStageSurface(filter_data *tf, uint32_t &width, uint32_t &height)
{

	if (!obs_source_enabled(tf->source)) {
		return false;
	}

	obs_source_t *target = obs_filter_get_target(tf->source);
	if (!target) {
		return false;
	}
	width = obs_source_get_base_width(target);
	height = obs_source_get_base_height(target);
	if (width == 0 || height == 0) {
		return false;
	}
	gs_texrender_reset(tf->texrender);
	if (!gs_texrender_begin(tf->texrender, width, height)) {
		return false;
	}
	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f,
		 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(tf->texrender);

	if (tf->stagesurface) {
		uint32_t stagesurf_width = gs_stagesurface_get_width(tf->stagesurface);
		uint32_t stagesurf_height = gs_stagesurface_get_height(tf->stagesurface);
		if (stagesurf_width != width || stagesurf_height != height) {
			gs_stagesurface_destroy(tf->stagesurface);
			tf->stagesurface = nullptr;
		}
	}
	if (!tf->stagesurface) {
		tf->stagesurface = gs_stagesurface_create(width, height, GS_BGRA);
	}
	gs_stage_texture(tf->stagesurface, gs_texrender_get_texture(tf->texrender));
	uint8_t *video_data;
	uint32_t linesize;
	if (!gs_stagesurface_map(tf->stagesurface, &video_data, &linesize)) {
		return false;
	}
	{
		std::lock_guard<std::mutex> lock(tf->inputBGRALock);
		cv::Mat ownedBGRA(static_cast<int>(height), static_cast<int>(width), CV_8UC4);
		const uint32_t row_size = width * 4u;
		for (uint32_t row = 0; row < height; ++row) {
			const uint8_t *src_row = video_data + row * linesize;
			uint8_t *dst_row = ownedBGRA.ptr<uint8_t>(static_cast<int>(row));
			std::memcpy(dst_row, src_row, row_size);
		}
		tf->inputBGRA = std::move(ownedBGRA);
	}
	gs_stagesurface_unmap(tf->stagesurface);
	return true;
}

gs_texture_t *blur_image(struct filter_data *tf, uint32_t width, uint32_t height,
			 gs_texture_t *alphaTexture)
{
	gs_texture_t *blurredTexture = ensure_effect_work_texture(tf, width, height);
	if (!blurredTexture) {
		obs_log(LOG_ERROR, "Failed to create blur work texture");
		return nullptr;
	}
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	if (tf->kawaseBlurEffect == nullptr) {
		obs_log(LOG_ERROR, "tf->kawaseBlurEffect is null");
		return blurredTexture;
	}
	gs_eparam_t *image = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "image");
	gs_eparam_t *xOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "xOffset");
	gs_eparam_t *yOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "yOffset");
	gs_eparam_t *mask = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "focalmask");

	for (int i = 0; i < (int)tf->maskingBlurRadius; i++) {
		gs_texrender_reset(tf->texrender);
		if (!gs_texrender_begin(tf->texrender, width, height)) {
			obs_log(LOG_INFO, "Could not open background blur texrender!");
			return blurredTexture;
		}

		gs_effect_set_texture(image, blurredTexture);
		if (alphaTexture != nullptr) {
			gs_effect_set_texture(mask, alphaTexture);
		}
		gs_effect_set_float(xOffset, ((float)i + 0.5f) / (float)width);
		gs_effect_set_float(yOffset, ((float)i + 0.5f) / (float)height);

		struct vec4 background;
		vec4_zero(&background);
		gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
		gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f,
			 100.0f);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

		while (gs_effect_loop(tf->kawaseBlurEffect,
				      (alphaTexture == nullptr) ? "Draw" : "DrawMaskAware")) {
			gs_draw_sprite(blurredTexture, 0, width, height);
		}
		gs_blend_state_pop();
		gs_texrender_end(tf->texrender);
		gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	}
	return blurredTexture;
}

gs_texture_t *pixelate_image(struct filter_data *tf, uint32_t width, uint32_t height,
			     gs_texture_t *alphaTexture, float pixelateRadius)
{
	gs_texture_t *blurredTexture = ensure_effect_work_texture(tf, width, height);
	if (!blurredTexture) {
		obs_log(LOG_ERROR, "Failed to create pixelate work texture");
		return nullptr;
	}
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	if (tf->pixelateEffect == nullptr) {
		obs_log(LOG_ERROR, "tf->pixelateEffect is null");
		return blurredTexture;
	}
	gs_eparam_t *image = gs_effect_get_param_by_name(tf->pixelateEffect, "image");
	gs_eparam_t *mask = gs_effect_get_param_by_name(tf->pixelateEffect, "focalmask");
	gs_eparam_t *pixel_size = gs_effect_get_param_by_name(tf->pixelateEffect, "pixel_size");
	gs_eparam_t *tex_size = gs_effect_get_param_by_name(tf->pixelateEffect, "tex_size");

	gs_texrender_reset(tf->texrender);
	if (!gs_texrender_begin(tf->texrender, width, height)) {
		obs_log(LOG_INFO, "Could not open background blur texrender!");
		return blurredTexture;
	}

	gs_effect_set_texture(image, blurredTexture);
	if (alphaTexture != nullptr) {
		gs_effect_set_texture(mask, alphaTexture);
	}
	gs_effect_set_float(pixel_size, pixelateRadius);
	vec2 texsize_vec;
	vec2_set(&texsize_vec, (float)width, (float)height);
	gs_effect_set_vec2(tex_size, &texsize_vec);

	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f,
		 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

	while (gs_effect_loop(tf->pixelateEffect, "Draw")) {
		gs_draw_sprite(blurredTexture, 0, width, height);
	}
	gs_blend_state_pop();
	gs_texrender_end(tf->texrender);
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));

	return blurredTexture;
}
