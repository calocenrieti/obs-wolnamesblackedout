#ifndef CONSTS_H
#define CONSTS_H

const char *const USEGPU_CPU = "cpu";
const char *const USEGPU_DML = "dml";
const char *const USEGPU_CUDA = "cuda";
const char *const USEGPU_TENSORRT = "tensorrt";
const char *const USEGPU_COREML = "coreml";

const char *const KAWASE_BLUR_EFFECT_PATH = "effects/kawase_blur.effect";
const char *const MASKING_EFFECT_PATH = "effects/masking.effect";
const char *const PIXELATE_EFFECT_PATH = "effects/pixelate.effect";

const char *const PLUGIN_INFO_TEMPLATE =
	"<center><a href=\"https://github.com/calocenrieti/obs-wolnamesblackedout\">OBS WoLNamesBlackedOut Plugin</a> (%1) by "
	"<a href=\"https://github.com/calocenrieti\">Calocen Rieti</a><br>"
	"<a href=\"https://blog.calocenrieti.com/blog/obs-wolnamesblackedout/\">Support & Follow</a></center>";
const char *const PLUGIN_INFO_TEMPLATE_UPDATE_AVAILABLE =
	"<center><a href=\"https://github.com/calocenrieti/obs-wolnamesblackedout/releases\">🚀 Update available! (%1)</a></center>";

#endif /* CONSTS_H */
