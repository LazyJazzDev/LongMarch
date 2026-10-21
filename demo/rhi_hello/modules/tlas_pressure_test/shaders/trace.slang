struct CameraInfo {
  float4x4 screen_to_camera;
  float4x4 camera_to_world;
};

RHI_ACCELERATION_STRUCTURE(0, 0, scene);
RHI_RW_TEXTURE2D(0, 1, float4, output);
RHI_UNIFORM(0, 2, CameraInfo, camera);
RHI_STRUCTURED_BUFFER(0, 3, float4, colors);

struct Hit {
  float distance;
  float3 normal;
  uint object;
};

bool TraceScene(RayDesc ray, uint flags, out Hit hit) {
  RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
  query.TraceRayInline(scene, flags, 0xff, ray);
  while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
      float3 o = query.CandidateObjectRayOrigin();
      float3 d = query.CandidateObjectRayDirection();
      float a = dot(d, d);
      float b = dot(o, d);
      float discriminant = b * b - a * (dot(o, o) - .25);
      if (discriminant >= 0) {
        float root = sqrt(discriminant);
        float t = (-b - root) / a;
        if (t < ray.TMin)
          t = (-b + root) / a;
        float closest = query.CommittedStatus() == COMMITTED_NOTHING ? ray.TMax : query.CommittedRayT();
        if (t >= ray.TMin && t <= closest)
          query.CommitProceduralPrimitiveHit(t);
      }
    }
  }
  hit = (Hit)0;
  if (query.CommittedStatus() == COMMITTED_NOTHING)
    return false;
  hit.distance = query.CommittedRayT();
  hit.object = query.CommittedInstanceID();
  if (hit.object == 1000) {
    hit.normal = float3(0, 1, 0);
  } else {
    float3 p = query.CommittedObjectRayOrigin() + hit.distance * query.CommittedObjectRayDirection();
    if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
      hit.normal = normalize(p);
    } else {
      float3 q = abs(p);
      hit.normal = q.x > q.y && q.x > q.z ? float3(sign(p.x), 0, 0)
                   : q.y > q.z            ? float3(0, sign(p.y), 0)
                                          : float3(0, 0, sign(p.z));
    }
  }
  return true;
}

[numthreads(8, 8, 1)] void CSMain(uint3 pixel
                                  : SV_DispatchThreadID) {
  uint width, height;
  output.GetDimensions(width, height);
  if (pixel.x >= width || pixel.y >= height)
    return;
  float2 uv = (float2(pixel.xy) + .5) / float2(width, height);
  uv.y = 1 - uv.y;
  float4 target = mul(camera.screen_to_camera, float4(uv * 2 - 1, 1, 1));
  RayDesc ray;
  ray.Origin = mul(camera.camera_to_world, float4(0, 0, 0, 1)).xyz;
  ray.Direction = normalize(mul(camera.camera_to_world, float4(target.xyz, 0)).xyz);
  ray.TMin = .001;
  ray.TMax = 200;
  float3 result = float3(.6, .7, .8);
  Hit hit;
  if (TraceScene(ray, RAY_FLAG_NONE, hit)) {
    float3 p = ray.Origin + hit.distance * ray.Direction;
    float3 light = normalize(float3(-.6, 1, .4));
    float diffuse = max(dot(hit.normal, light), 0);
    float visibility = 1;
    if (diffuse > 0) {
      RayDesc shadow;
      shadow.Origin = p + hit.normal * .003;
      shadow.Direction = light;
      shadow.TMin = .001;
      shadow.TMax = 200;
      Hit blocker;
      if (TraceScene(shadow, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, blocker))
        visibility = 0;
    }
    float3 albedo = colors[hit.object].rgb;
    if (hit.object == 1000) {
      float checker = fmod(abs(floor(p.x) + floor(p.z)), 2);
      albedo *= lerp(.8, 1, checker);
    }
    result = albedo * (.18 + .82 * diffuse * visibility);
  }
  output[pixel.xy] = float4(result, 1);
}
