#pragma once

#include <memory>
#include "ort-model/types.hpp"

namespace ByteTrack {

class Track {
public:
    using DetectionPtr = std::shared_ptr<class Detection>;

    Track(const DetectionPtr &det) {
        if (det) {
            const Object &o = det->getObject();
            obj_ = o;
            // unseen_frames_ = 0;
        }
        track_id_ = 0;
    }

    void update(const DetectionPtr &det) {
        if (!det) return;
        const Object &o = det->getObject();
        obj_.rect = o.rect;
        obj_.prob = o.prob;
        obj_.label = o.label;
        // unseen_frames_ = 0;
    }

    void set_track_id(size_t id) { track_id_ = id; obj_.id = id; }
    size_t track_id() const { return track_id_; }

    const Object &getObject() const { return obj_; }

private:
    Object obj_;
    // uint64_t unseen_frames_ = 0;
    size_t track_id_ = 0;
};

} // namespace ByteTrack
