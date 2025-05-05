#pragma once
#include "grassland/math/math_aabb.h"
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

template <class Real>
LM_DEVICE_FUNC Real DistancePointAABB(const Vector3<Real> &p,
                                      const Vector3<Real> &lower_bound,
                                      const Vector3<Real> &upper_bound);

template <class Real>
LM_DEVICE_FUNC Real DistancePointAABB(const Vector3<Real> &p, const AxisAlignedBoundingBox3<Real> &aabb);

template <class Real>
LM_DEVICE_FUNC Real AnyHitRayAABB(const Vector3<Real> &origin,
                                  const Vector3<Real> &direction,
                                  const Vector3<Real> &lower_bound,
                                  const Vector3<Real> &upper_bound,
                                  Real t_min = Eps<Real>(),
                                  Real t_max = 1.0 / Eps<Real>());

template <class Real>
LM_DEVICE_FUNC Real AnyHitRayAABB(const Vector3<Real> &origin,
                                  const Vector3<Real> &direction,
                                  const AxisAlignedBoundingBox3<Real> &aabb,
                                  Real t_min = Eps<Real>(),
                                  Real t_max = 1.0 / Eps<Real>());

template <class Real>
LM_DEVICE_FUNC bool IsSegmentTriangleIntersect(const Vector3<Real> &s0,
                                               const Vector3<Real> &s1,
                                               const Vector3<Real> &t0,
                                               const Vector3<Real> &t1,
                                               const Vector3<Real> &t2);

template <class Real>
LM_DEVICE_FUNC Real AnyHitRayTriangle(const Vector3<Real> &origin,
                                      const Vector3<Real> &direction,
                                      const Vector3<Real> &v0,
                                      const Vector3<Real> &v1,
                                      const Vector3<Real> &v2,
                                      Real t_min = Eps<Real>(),
                                      Real t_max = 1.0 / Eps<Real>());

}  // namespace grassland
