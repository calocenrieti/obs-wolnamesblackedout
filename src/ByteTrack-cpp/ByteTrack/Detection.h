#pragma once

#include "ort-model/types.hpp"
#include "ByteTrack/Rect.h"

namespace ByteTrack {

class Detection {
public:
    Detection(const Object &obj)
        : obj_(obj) {}

    // Return rectangle in Tlwh format expected by byte_track
    byte_track::TlwhRect rect() const {
        // TlwhRect expects (top, left, width, height)
        float top = obj_.rect.y;
        float left = obj_.rect.x;
        float width = obj_.rect.width;
        float height = obj_.rect.height;
        return byte_track::TlwhRect(top, left, width, height);
    }

    float score() const { return obj_.prob; }

    const Object &getObject() const { return obj_; }

private:
    Object obj_;
};

} // namespace ByteTrack
