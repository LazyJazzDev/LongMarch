#pragma once
#include "principled_util.hlsli"

#define CLOSURE_COUNT 6

class PrincipledMaterial {
#include "principled_bsdf.hlsli"
#include "principled_diffuse.hlsli"
#include "principled_microfacet.hlsli"
#include "principled_microfacet_clearcoat.hlsli"
#include "principled_microfacet_fresnel.hlsli"
#include "principled_microfacet_refraction.hlsli"
#include "principled_sheen.hlsli"

  HitRecord hit_record;
  float3 omega_v;
  PrincipledDiffuseBsdf diffuse_closure;
  FresnelBsdf microfacet_closure;
  FresnelBsdf microfacet_bsdf_reflect_closure;
  RefractionBsdf microfacet_bsdf_refract_closure;
  ClearcoatBsdf microfacet_clearcoat_closure;
  PrincipledSheenBsdf sheen_closure;

  float3 base_color;

  float3 subsurface_color;
  float subsurface;

  float3 subsurface_radius;
  float metallic;

  float specular;
  float specular_tint;
  float roughness;
  float anisotropic;

  float anisotropic_rotation;
  float sheen;
  float sheen_tint;
  float clearcoat;

  float clearcoat_roughness;
  float ior;
  float transmission;
  float transmission_roughness;
};
