#pragma once
#include "grassland/math/math_util.h"

namespace grassland {

template <class Real>
LM_DEVICE_FUNC Real DistancePointPoint(const Vector3<Real> &p0, const Vector3<Real> &p1);

template <class Real>
LM_DEVICE_FUNC Real
DistancePointLine(const Vector3<Real> &p, const Vector3<Real> &s0, const Vector3<Real> &s1, Real &t);

template <class Real>
LM_DEVICE_FUNC Real DistancePointLine(const Vector3<Real> &p, const Vector3<Real> &s0, const Vector3<Real> &s1);

template <class Real>
LM_DEVICE_FUNC Real
DistancePointSegment(const Vector3<Real> &p, const Vector3<Real> &s0, const Vector3<Real> &s1, Real &t);

template <class Real>
LM_DEVICE_FUNC Real DistancePointSegment(const Vector3<Real> &p, const Vector3<Real> &s0, const Vector3<Real> &s1);

template <class Real>
LM_DEVICE_FUNC Real DistancePointPlane(const Vector3<Real> &p,
                                       const Vector3<Real> &v0,
                                       const Vector3<Real> &v1,
                                       const Vector3<Real> &v2,
                                       Real &u,
                                       Real &v);

template <class Real>
LM_DEVICE_FUNC Real
DistancePointPlane(const Vector3<Real> &p, const Vector3<Real> &v0, const Vector3<Real> &v1, const Vector3<Real> &v2);

// Compute the distance between a point and a triangle. Return -1 if the point's projection is outside the triangle.
template <class Real>
LM_DEVICE_FUNC Real DistancePointTriangleProjection(const Vector3<Real> &p,
                                                    const Vector3<Real> &v0,
                                                    const Vector3<Real> &v1,
                                                    const Vector3<Real> &v2);

template <class Real>
LM_DEVICE_FUNC Real DistancePointTriangle(const Vector3<Real> &p,
                                          const Vector3<Real> &v0,
                                          const Vector3<Real> &v1,
                                          const Vector3<Real> &v2,
                                          Real &u,
                                          Real &v);

template <class Real>
LM_DEVICE_FUNC Real DistancePointTriangle(const Vector3<Real> &p,
                                          const Vector3<Real> &v0,
                                          const Vector3<Real> &v1,
                                          const Vector3<Real> &v2);

template <class Real>
LM_DEVICE_FUNC Real DistanceLineLine(const Vector3<Real> &p0,
                                     const Vector3<Real> &p1,
                                     const Vector3<Real> &q0,
                                     const Vector3<Real> &q1,
                                     Real &u,
                                     Real &v);

template <class Real>
LM_DEVICE_FUNC Real
DistanceLineLine(const Vector3<Real> &p0, const Vector3<Real> &p1, const Vector3<Real> &q0, const Vector3<Real> &q1);

template <class Real>
LM_DEVICE_FUNC Real DistanceSegmentSegment(const Vector3<Real> &p0,
                                           const Vector3<Real> &p1,
                                           const Vector3<Real> &q0,
                                           const Vector3<Real> &q1,
                                           Real &u,
                                           Real &v);

template <class Real>
LM_DEVICE_FUNC Real DistanceSegmentSegment(const Vector3<Real> &p0,
                                           const Vector3<Real> &p1,
                                           const Vector3<Real> &q0,
                                           const Vector3<Real> &q1);

}  // namespace grassland
