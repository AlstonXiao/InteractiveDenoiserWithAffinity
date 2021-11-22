// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "gdt/random/random.h"

using namespace osc;

#define NUM_LIGHT_SAMPLES 16
#define MAX_TRACE_DEPTH 12
#define RR_DEPTH 3

#define MATTE 0 
#define PLASTIC 1
#define SUBSTRATE 2
#define METAL 3
#define UBER 4
#define MIRROR 5

namespace osc {

  typedef gdt::LCG<16> Random;
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  /*! per-ray data now captures random number generator, so programs
      can access RNG state */
  struct PRD {
    Random random;
    vec3f  pixelColor;
    vec3f  pixelNormal;
    vec3f  pixelAlbedo;

    vec3f pixelScolor;
    vec3f pixelDcolor;
    float pixeldepth;
    float pixelroughness;
    bool  pixelspecular_bounce;
    bool  pixelmetallic;
    bool  pixelemissive;

    int   recursion;
    vec3f contribution_to_pixel;
  };
  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  //------------------------------------------------------------------------------
  // useful math function for shader
  //------------------------------------------------------------------------------
  template <typename T, typename U, typename V>
  __forceinline__ __device__ T Clamp(T val, U low, V high) {
      if (val < low)
          return low;
      else if (val > high)
          return high;
      else
          return val;
  }
  // BSDF Inline Functions
  __forceinline__ __device__ float CosTheta(const vec3f& w) { return w.z; }
  __forceinline__ __device__ float Cos2Theta(const vec3f& w) { return w.z * w.z; }
  __forceinline__ __device__ float AbsCosTheta(const vec3f& w) { return abs(w.z); }
  __forceinline__ __device__ float Sin2Theta(const vec3f& w) {
      return fmax((float)0, (float)1 - Cos2Theta(w));
  }

  __forceinline__ __device__ float SinTheta(const vec3f& w) { return sqrt(Sin2Theta(w)); }

  __forceinline__ __device__ float TanTheta(const vec3f& w) { return SinTheta(w) / CosTheta(w); }

  __forceinline__ __device__ float Tan2Theta(const vec3f& w) {
      return Sin2Theta(w) / Cos2Theta(w);
  }

  __forceinline__ __device__ float CosPhi(const vec3f& w) {
      float sinTheta = SinTheta(w);
      return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1, 1);
  }

  __forceinline__ __device__ float SinPhi(const vec3f& w) {
      float sinTheta = SinTheta(w);
      return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1, 1);
  }

  __forceinline__ __device__ float Cos2Phi(const vec3f& w) { return CosPhi(w) * CosPhi(w); }

  __forceinline__ __device__ float Sin2Phi(const vec3f& w) { return SinPhi(w) * SinPhi(w); }

  __forceinline__ __device__ float CosDPhi(const vec3f& wa, const vec3f& wb) {
      float waxy = wa.x * wa.x + wa.y * wa.y;
      float wbxy = wb.x * wb.x + wb.y * wb.y;
      if (waxy == 0 || wbxy == 0)
          return 1;
      return Clamp((wa.x * wb.x + wa.y * wb.y) / sqrt(waxy * wbxy), -1, 1);
  }

  __device__ __inline__ vec3f fmaxf(const vec3f& a, const vec3f& b) {
      return vec3f(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
  }

  __device__ __inline__ vec3f fminf(const vec3f& a, const vec3f& b) {
      return vec3f(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
  }
  __device__ __inline__ float fmaxf(const vec3f& a) {
      return fmax(a.x, fmax(a.y, a.z));
  }

  __device__ __inline__ float fminf(const vec3f& a) {
      return fmin(a.x, fmin(a.y, a.z));
  }
  __device__ __inline__ vec3f Pow(const vec3f& v, const float& a) {
      return vec3f(pow(v.x, a), pow(v.y, a), pow(v.z, a));
  }
  __device__ __inline__ vec3f Sqrt(const vec3f& v) {
      return vec3f(sqrt(v.x), sqrt(v.y), sqrt(v.z));
  }
  __device__ __inline__  vec3f Faceforward(const vec3f& v, const vec3f& v2) {
      return (dot(v, v2) < 0.f) ? -v : v;
  }
  __device__ __inline__ vec3f sample_hemisphere_dir(const float& z1, const float& z2, const vec3f& normal) {
      const float radius = sqrtf(z1);
      const float theta = 2.f * M_PI * z2;
      float x = radius * cosf(theta);
      float y = radius * sinf(theta);
      float z = sqrtf(fmax(0.f, 1.f - x * x - y * y));
      vec3f binormal = vec3f(0);

      // Prevent normal = (0, 0, 1)
      if (fabs(normal.x) > fabs(normal.z)) {
          binormal.x = -normal.y;
          binormal.y = normal.x;
          binormal.z = 0;
      }
      else {
          binormal.x = 0;
          binormal.y = -normal.z;
          binormal.z = normal.y;
      }

      // float3 binormal = make_float3(-normal.y, normal.x, 0);
      binormal = normalize(binormal);
      vec3f tangent = cross(normal, binormal);
      return normalize(x * tangent + y * binormal + z * normal);
  }


  __device__ __inline__ float D(const vec3f& wh, const float& roughness2) {
      float tan2Theta = Tan2Theta(wh);
      if ((tan2Theta) > INFINITY) return 0.;
      const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
      float e =
          (Cos2Phi(wh) / (roughness2) + Sin2Phi(wh) / (roughness2)) *
          tan2Theta;
      return 1 / (M_PI * roughness2 * cos4Theta * (1 + e) * (1 + e));
  }
  __device__ __inline__ vec3f SchlickFresnel(float cosTheta, vec3f specular) {
      auto pow5 = [](float v) { return (v * v) * (v * v) * v; };
      return specular + pow5(1 - cosTheta) * (vec3f(1.) - specular);
  }
  __device__ __inline__ float Lambda(const vec3f& w, const float& roughness2) {
      float absTanTheta = abs(TanTheta(w));
      if ((absTanTheta) > INFINITY) return 0.;
      // Compute _alpha_ for direction _w_
      float alpha =
          sqrt(Cos2Phi(w) * roughness2 + Sin2Phi(w) * roughness2);
      float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
      return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
  }
  __device__ __inline__ float G(const vec3f& wo, const vec3f& wi, const float& roughness2) {
      return 1 / (1 + Lambda(wo, roughness2) + Lambda(wi, roughness2));
  }
  __device__ __inline__ float FrDielectric(float cosThetaI, float etaI, float etaT) {
      cosThetaI = Clamp(cosThetaI, -1, 1);
      // Potentially swap indices of refraction
      bool entering = cosThetaI > 0.f;
      if (!entering) {
          float temp = etaI;
          etaI = etaT;
          etaT = temp;
          cosThetaI = abs(cosThetaI);
      }

      // Compute _cosThetaT_ using Snell's law
      float sinThetaI = sqrt(fmax((float)0, 1 - cosThetaI * cosThetaI));
      float sinThetaT = etaI / etaT * sinThetaI;

      // Handle total internal reflection
      if (sinThetaT >= 1) return 1;
      float cosThetaT = sqrt(fmax((float)0, 1 - sinThetaT * sinThetaT));
      float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
          ((etaT * cosThetaI) + (etaI * cosThetaT));
      float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
          ((etaI * cosThetaI) + (etaT * cosThetaT));
      return (Rparl * Rparl + Rperp * Rperp) / 2;
  }


  __device__ __inline__ vec3f FrConductor(float cosThetaI, const vec3f& etai,
      const vec3f& etat, const vec3f& k) {
      cosThetaI = abs(cosThetaI);
      cosThetaI = Clamp(cosThetaI, -1, 1);
      vec3f eta = etat / etai;
      vec3f etak = k / etai;

      float cosThetaI2 = cosThetaI * cosThetaI;
      float sinThetaI2 = 1. - cosThetaI2;
      vec3f eta2 = eta * eta;
      vec3f etak2 = etak * etak;

      vec3f t0 = eta2 - etak2 - sinThetaI2;
      vec3f a2plusb2 = Sqrt(t0 * t0 + vec3f(4.0) * eta2 * etak2);
      vec3f t1 = a2plusb2 + cosThetaI2;
      vec3f a = Sqrt(0.5f * (a2plusb2 + t0));
      vec3f t2 = (float)2 * cosThetaI * a;
      vec3f Rs = (t1 - t2) / (t1 + t2);

      vec3f t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
      vec3f t4 = t2 * sinThetaI2;
      vec3f Rp = Rs * (t3 - t4) / (t3 + t4);

      return vec3f(0.5) * (Rp + Rs);
  }

  __device__ __inline__ vec3f WorldToLocal(const vec3f& v, const vec3f& ns) {
      vec3f ss;

      // Prevent normal = (0, 0, 1)
      if (abs(ns.x) > abs(ns.z)) {
          ss.x = -ns.y;
          ss.y = ns.x;
          ss.z = 0;
      }
      else {
          ss.x = 0;
          ss.y = -ns.z;
          ss.z = ns.y;
      }
      ss = normalize(ss);
      vec3f ts = cross(ns, ss);
      ts = normalize(ts);
      return normalize(vec3f(dot(v, ss), dot(v, ts), dot(v, ns)));
  }


  __device__ __inline__ vec3f WorldToLocal(const vec3f& v, const vec3f& ns, const vec3f& ss) {
      if (dot(ss, ns) > 1e-3) printf("error\n");
      vec3f ts = cross(ns, ss);
      ts = normalize(ts);
      return normalize(vec3f(dot(v, ss), dot(v, ts), dot(v, ns)));
  }
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__shadow()
  {
      // we didn't hit anything, so the light is visible
      vec3f& prd = *(vec3f*)getPRD<vec3f>();
      prd = vec3f(0.5f);
  }
  
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const vec3f &A     = sbtData.vertex[index.x];
    const vec3f &B     = sbtData.vertex[index.y];
    const vec3f &C     = sbtData.vertex[index.z];
    vec3f Ng = cross(B-A,C-A);
    vec3f Ns = (sbtData.normal)
      ? ((1.f-u-v) * sbtData.normal[index.x]
         +       u * sbtData.normal[index.y]
         +       v * sbtData.normal[index.z])
      : Ng;
    
    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const vec3f rayDir = optixGetWorldRayDirection();
    float depth = optixGetRayTmax();

    if (dot(rayDir,Ng) > 0.f) Ng = -Ng;
    Ng = normalize(Ng);
    
    if (dot(Ng,Ns) < 0.f)
      Ns -= 2.f*dot(Ng,Ns)*Ng;
    Ns = normalize(Ns);

    if (dot(rayDir, Ns) > 0.f) Ns = -Ns;

    const vec3f surfPos
        = (1.f - u - v) * sbtData.vertex[index.x]
        + u * sbtData.vertex[index.y]
        + v * sbtData.vertex[index.z];

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    vec3f diffuseColor = sbtData.kd;
    if (sbtData.hasKdTexture && sbtData.texcoord) {
        const vec2f tc
            = (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        vec4f fromTexture = tex2D<float4>(sbtData.kdTexture, tc.x, tc.y);
        diffuseColor *= (vec3f)fromTexture;
    }

    vec3f specularColor = sbtData.ks;
    if (sbtData.hasKsTexture && sbtData.texcoord) {
        const vec2f tc
            = (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        vec4f fromTexture = tex2D<float4>(sbtData.ksTexture, tc.x, tc.y);
        specularColor *= (vec3f)fromTexture;
    }

    vec3f pixelColorD = vec3f(0);
    vec3f pixelColorS = vec3f(0);

    int mattype = sbtData.type;

    // mirror is a special case here
    if (mattype == MIRROR) {
        if (prd.recursion < MAX_TRACE_DEPTH) {
            prd.recursion++;
            uint32_t u0, u1;
            packPointer(&prd, u0, u1);
            // calculate reflect direction

            vec3f reflect_dir = rayDir - vec3f(2) * (dot(rayDir, Ns) * Ns);

            optixTrace(optixLaunchParams.traversable,
                surfPos + 1e-3f * Ng,
                reflect_dir,
                1e-3f,      // tmin
                1e20f,  // tmax
                0.0f,       // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE,            // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                RADIANCE_RAY_TYPE,            // missSBTIndex 
                u0, u1);

            // These will use the actually reflected position
            // prd.pixelNormal = Ns;
            // prd.pixelAlbedo = vec3f(0);

            // add the depth from camera to the mirror
            prd.pixeldepth += depth;
            // prd.pixelmetallic = false;
            prd.pixelspecular_bounce = true;
            // prd.pixelroughness = 0;

            // These will not change, but will apply the attenuation 
            prd.pixelScolor *= specularColor;
            prd.pixelDcolor *= specularColor;
            // prd.pixelColor *= specularColor;
            return;
        }
        else {
            // Other information does not needed at this depth
            // If iterative solution, this might not be needed (cant reflect a mirror 17 times)
            prd.pixelScolor = specularColor;
            prd.pixelDcolor = specularColor;
            prd.pixelColor = specularColor;
            return;
        }
    }
    
    const int numLightSamples = NUM_LIGHT_SAMPLES;
    for (int lightID = 0; lightID < optixLaunchParams.numLights; lightID++) {
        for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++) {
            int rowID = lightSampleID / 4;
            int columnID = lightSampleID % 4;
            // produce random light sample
            const vec3f lightPos
                = optixLaunchParams.lights[lightID].origin
                + prd.random() / 4.0f * optixLaunchParams.lights[lightID].du
                + prd.random() / 4.0f * optixLaunchParams.lights[lightID].dv
                + (float)rowID / 4.0f * optixLaunchParams.lights[lightID].du
                + (float)columnID / 4.0f * optixLaunchParams.lights[lightID].dv;
            vec3f lightDir = lightPos - surfPos;
            float lightDist = gdt::length(lightDir);
            lightDir = normalize(lightDir);

            // trace shadow ray:
            const float NdotL = dot(lightDir, Ns);
            if (NdotL >= 0.f) {
                vec3f lightVisibility = 0.f;
                // the values we store the PRD pointer in:
                uint32_t u0, u1;
                packPointer(&lightVisibility, u0, u1);
                optixTrace(optixLaunchParams.traversable,
                    surfPos + 1e-3f * Ng,
                    lightDir,
                    1e-3f,      // tmin
                    lightDist * (1.f - 1e-3f),  // tmax
                    0.0f,       // rayTime
                    OptixVisibilityMask(255),
                    // For shadow rays: skip any/closest hit shaders and terminate on first
                    // intersection with anything. The miss shader is used to mark if the
                    // light was visible.
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT
                    | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                    | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                    SHADOW_RAY_TYPE,            // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride
                    SHADOW_RAY_TYPE,            // missSBTIndex 
                    u0, u1);
                
                vec3f incoming_light_ray = optixLaunchParams.lights[lightID].power * lightVisibility * NdotL / (lightDist * lightDist);
                // the light has contribution to the surface
                if (fmaxf(incoming_light_ray) > 1e-3f) {
                    vec3f wi = normalize(WorldToLocal(normalize(lightDir), Ns));
                    vec3f wo = normalize(WorldToLocal(normalize(-rayDir), Ns)); 
                    float InvPi = 1 / M_PI;

                    if (mattype == MATTE) {
                        float sigma2 = sbtData.roughness_square;
                        if (sigma2 == 0) {
                            pixelColorD += incoming_light_ray * diffuseColor / M_PI / numLightSamples;
                        }
                        else {
                            float A = 1 - sigma2 / (2*(sigma2 + 0.33));
                            float B = 0.45 * sigma2 / (sigma2 + 0.09);
                            float sinThetaI = SinTheta(wi);
                            float sinThetaO = SinTheta(wo);
                            // Compute cosine term of Oren-Nayar model
                            float maxCos = 0;
                            if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
                                float sinPhiI = SinPhi(wi), cosPhiI = CosPhi(wi);
                                float sinPhiO = SinPhi(wo), cosPhiO = CosPhi(wo);
                                float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
                                maxCos = fmax((float)0, dCos);
                            }

                            // Compute sine and tangent terms of Oren-Nayar model
                            float sinAlpha, tanBeta;
                            if (AbsCosTheta(wi) > AbsCosTheta(wo)) {
                                sinAlpha = sinThetaO;
                                tanBeta = sinThetaI / AbsCosTheta(wi);
                            }
                            else {
                                sinAlpha = sinThetaI;
                                tanBeta = sinThetaO / AbsCosTheta(wo);
                            }
                            pixelColorD += incoming_light_ray * diffuseColor * InvPi * (A + B * maxCos * sinAlpha * tanBeta) / numLightSamples;
                        }
                    }
                    else if (mattype == PLASTIC) {
                        pixelColorD += incoming_light_ray * diffuseColor / M_PI / numLightSamples;
                        float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
                        vec3f wh = wi + wo;
                        // Handle degenerate cases for microfacet reflection
                        if (cosThetaI == 0 || cosThetaO == 0 || (wh.x == 0 && wh.y == 0 && wh.z == 0)) {
                            pixelColorS += vec3f(0);
                        }
                        else {
                            wh = normalize(wh);
                            // For the Fresnel call, make sure that wh is in the same hemisphere
                            // as the surface normal, so that TIR is handled correctly.
                            vec3f F = FrDielectric(dot(wi, Faceforward(wh, vec3f(0, 0, 1))), 1.5, 1);
                            pixelColorS += specularColor * D(wh, sbtData.roughness_square) * G(wo, wi, sbtData.roughness_square) * F /
                                (4 * cosThetaI * cosThetaO) * incoming_light_ray / numLightSamples;

                        }
                    }
                    else if (mattype == SUBSTRATE) {
                        auto pow5 = [](float v) { return (v * v) * (v * v) * v; };
                        vec3f diffuse = (28.f / (23.f * M_PI)) * diffuseColor * (vec3f(1.f) - specularColor) *
                            (1 - pow5(1 - .5f * AbsCosTheta(wi))) *
                            (1 - pow5(1 - .5f * AbsCosTheta(wo)));
                        vec3f wh = wi + wo;
                        if (wh.x == 0 && wh.y == 0 && wh.z == 0) {
                            pixelColorD += vec3f(0);
                            pixelColorS += vec3f(0);
                        }
                        else {
                            wh = normalize(wh);
                            vec3f specular =
                                D(wh, sbtData.roughness_square) /
                                (4* abs(dot(wi, wh))* fmax(AbsCosTheta(wi), AbsCosTheta(wo))) *
                                SchlickFresnel(dot(wi, wh), specularColor);
                            pixelColorD +=  diffuse * incoming_light_ray / numLightSamples ;
                            pixelColorS += specular* incoming_light_ray / numLightSamples;
                        }
                    }
                    else if (mattype == METAL) {
                        float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
                        vec3f wh = wi + wo;
                        // Handle degenerate cases for microfacet reflection
                        if (cosThetaI == 0 || cosThetaO == 0 || (wh.x == 0 && wh.y == 0 && wh.z == 0)) {
                            pixelColorS += vec3f(0);
                        }
                        else {
                            wh = normalize(wh);
                            // For the Fresnel call, make sure that wh is in the same hemisphere
                            // as the surface normal, so that TIR is handled correctly.
                            vec3f F = FrConductor(dot(wi, Faceforward(wh, vec3f(0, 0, 1))), vec3f(1.0), diffuseColor, specularColor);
                            pixelColorS += D(wh, sbtData.roughness_square) * G(wo, wi, sbtData.roughness_square) * F /
                                (4 * cosThetaI * cosThetaO)*incoming_light_ray / numLightSamples;

                        }
                    }
                    else if (mattype == UBER) {
                        pixelColorD += incoming_light_ray * diffuseColor / M_PI / numLightSamples;
                        float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
                        vec3f wh = wi + wo;
                        // Handle degenerate cases for microfacet reflection
                        if (cosThetaI == 0 || cosThetaO == 0 || (wh.x == 0 && wh.y == 0 && wh.z == 0)) {
                            pixelColorS += vec3f(0);
                        }
                        else {
                            wh = normalize(wh);
                            // For the Fresnel call, make sure that wh is in the same hemisphere
                            // as the surface normal, so that TIR is handled correctly.
                            vec3f F = FrDielectric(dot(wi, Faceforward(wh, vec3f(0, 0, 1))), 1.0, 1.5);
                            pixelColorS += specularColor * D(wh, sbtData.roughness_square) * G(wo, wi, sbtData.roughness_square) * F /
                            (4 * cosThetaI * cosThetaO) * incoming_light_ray / numLightSamples;

                        }
                    }

                } 
            }
        }
    }

    // recursive trace
    // simple random sampling
    vec3f next_dir = sample_hemisphere_dir(prd.random(), prd.random(), Ns);
    const float NdotL = dot(next_dir, Ns);
 
    // the light has contribution to the surface
    vec3f wi = normalize(WorldToLocal(normalize(next_dir), Ns));
    vec3f wo = normalize(WorldToLocal(normalize(-rayDir), Ns));
    
    vec3f Dcontribution = vec3f(0);
    vec3f Scontribution = vec3f(0);

    if (mattype == MATTE) {
        float sigma2 = sbtData.roughness_square;
        if (sigma2 == 0) {
            Dcontribution = diffuseColor / M_PI;
        }
        else {
            float A = 1 - sigma2 / (2 * (sigma2 + 0.33));
            float B = 0.45 * sigma2 / (sigma2 + 0.09);
            float sinThetaI = SinTheta(wi);
            float sinThetaO = SinTheta(wo);
            // Compute cosine term of Oren-Nayar model
            float maxCos = 0;
            if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
                float sinPhiI = SinPhi(wi), cosPhiI = CosPhi(wi);
                float sinPhiO = SinPhi(wo), cosPhiO = CosPhi(wo);
                float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
                maxCos = fmax((float)0, dCos);
            }

            // Compute sine and tangent terms of Oren-Nayar model
            float sinAlpha, tanBeta;
            if (AbsCosTheta(wi) > AbsCosTheta(wo)) {
                sinAlpha = sinThetaO;
                tanBeta = sinThetaI / AbsCosTheta(wi);
            }
            else {
                sinAlpha = sinThetaI;
                tanBeta = sinThetaO / AbsCosTheta(wo);
            }
            Dcontribution =  diffuseColor / M_PI * (A + B * maxCos * sinAlpha * tanBeta);
        }
    }
    else if (mattype == PLASTIC) {
        Dcontribution = diffuseColor / M_PI ;
        float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
        vec3f wh = wi + wo;
        // Handle degenerate cases for microfacet reflection
        if (cosThetaI == 0 || cosThetaO == 0 || (wh.x == 0 && wh.y == 0 && wh.z == 0)) {
            Scontribution += vec3f(0);
        }
        else {
            wh = normalize(wh);
            // For the Fresnel call, make sure that wh is in the same hemisphere
            // as the surface normal, so that TIR is handled correctly.
            vec3f F = FrDielectric(dot(wi, Faceforward(wh, vec3f(0, 0, 1))), 1.5, 1);
            Scontribution += specularColor * D(wh, sbtData.roughness_square) * G(wo, wi, sbtData.roughness_square) * F /
                (4 * cosThetaI * cosThetaO);

        }
    }
    else if (mattype == SUBSTRATE) {
        auto pow5 = [](float v) { return (v * v) * (v * v) * v; };
        vec3f diffuse = (28.f / (23.f * M_PI)) * diffuseColor * (vec3f(1.f) - specularColor) *
            (1 - pow5(1 - .5f * AbsCosTheta(wi))) *
            (1 - pow5(1 - .5f * AbsCosTheta(wo)));
        vec3f wh = wi + wo;
        if (wh.x == 0 && wh.y == 0 && wh.z == 0) {
            Dcontribution = vec3f(0);
            Scontribution = vec3f(0);
        }
        else {
            wh = normalize(wh);
            vec3f specular =
                D(wh, sbtData.roughness_square) /
                (4 * abs(dot(wi, wh)) * fmax(AbsCosTheta(wi), AbsCosTheta(wo))) *
                SchlickFresnel(dot(wi, wh), specularColor);
            Dcontribution = diffuse;
            Scontribution = specular;
        }
    }
    else if (mattype == METAL) {
        float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
        vec3f wh = wi + wo;
        // Handle degenerate cases for microfacet reflection
        if (cosThetaI == 0 || cosThetaO == 0 || (wh.x == 0 && wh.y == 0 && wh.z == 0)) {
            Scontribution = vec3f(0);
        }
        else {
            wh = normalize(wh);
            // For the Fresnel call, make sure that wh is in the same hemisphere
            // as the surface normal, so that TIR is handled correctly.
            vec3f F = FrConductor(dot(wi, Faceforward(wh, vec3f(0, 0, 1))), vec3f(1.0), diffuseColor, specularColor);
            Scontribution = D(wh, sbtData.roughness_square) * G(wo, wi, sbtData.roughness_square) * F /
                (4 * cosThetaI * cosThetaO);
            Scontribution = vec3f(Scontribution.x > 1? 1: Scontribution.x, Scontribution.y > 1 ? 1 : Scontribution.y, Scontribution.z > 1 ? 1 : Scontribution.z);

            // printf("%f %f %f\n", Scontribution.x, Scontribution.y, Scontribution.z);

        }
    }
    else if (mattype == UBER) {
        Dcontribution = diffuseColor / M_PI ;
        float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
        vec3f wh = wi + wo;
        // Handle degenerate cases for microfacet reflection
        if (cosThetaI == 0 || cosThetaO == 0 || (wh.x == 0 && wh.y == 0 && wh.z == 0)) {
            Scontribution = vec3f(0);
        }
        else {
            wh = normalize(wh);
            // For the Fresnel call, make sure that wh is in the same hemisphere
            // as the surface normal, so that TIR is handled correctly.
            vec3f F = FrDielectric(dot(wi, Faceforward(wh, vec3f(0, 0, 1))), 1.0, 1.5);
            Scontribution += specularColor * D(wh, sbtData.roughness_square) * G(wo, wi, sbtData.roughness_square) * F /
                (4 * cosThetaI * cosThetaO);

        }
    }
    Scontribution *= M_PI * 2;
    Dcontribution *= M_PI * 2;
    vec3f totalContribution = (Dcontribution + Scontribution) * prd.contribution_to_pixel;
    // russian roulette, based on the true contribution to pixel
    float p = fmax(totalContribution.x, fmax(totalContribution.y, totalContribution.z));
    if (prd.recursion > RR_DEPTH) {
        if (prd.random() > p || prd.recursion + 1 > MAX_TRACE_DEPTH) {
            prd.pixelColor = pixelColorD + pixelColorS;
            prd.pixelDcolor = pixelColorD;
            prd.pixelScolor = pixelColorS;
            return;
        }
        // Now they might be huge, but after the recursion, they will be lower down
        Dcontribution *= 1.0f / p;
        Scontribution *= 1.0f / p;
    }
    prd.contribution_to_pixel = totalContribution;
    prd.recursion++;
    uint32_t u0, u1;
    packPointer(&prd, u0, u1);
    optixTrace(optixLaunchParams.traversable,
        surfPos + 1e-3f * Ng,
        next_dir,
        1e-3f,      // tmin
        1e20f,  // tmax
        0.0f,       // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        RADIANCE_RAY_TYPE,            // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        RADIANCE_RAY_TYPE,            // missSBTIndex 
        u0, u1);



    vec3f incoming_light_ray = prd.pixelColor * NdotL;
    prd.pixelNormal = Ns;
    prd.pixelAlbedo = mattype == METAL  ? vec3f(1) : diffuseColor;

    prd.pixeldepth = depth;
    prd.pixelmetallic = mattype == METAL;
    prd.pixelspecular_bounce = mattype == METAL;
    prd.pixelroughness = sbtData.roughness_square;

    prd.pixelScolor = pixelColorS + Scontribution * incoming_light_ray;
    prd.pixelDcolor = pixelColorD + Dcontribution * incoming_light_ray;
    prd.pixelColor = prd.pixelScolor + prd.pixelDcolor;
    prd.pixelemissive = false;
  }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }

  extern "C" __global__ void __anyhit__shadow()
  { /*! not going to be used */ }
  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    PRD &prd = *getPRD<PRD>();
    // set to constant white as background color
    prd.pixelColor = vec3f(1.f, 1.f, 1.f);
    prd.pixelScolor = vec3f(0.f, 0.f, 0.f);
    prd.pixelDcolor = vec3f(1.f, 1.f, 1.f);

    prd.pixelNormal = -(vec3f)optixGetWorldRayDirection();
    prd.pixelAlbedo = vec3f(1.f, 1.f, 1.f);

    prd.pixeldepth = -1;
    prd.pixelmetallic = false;
    prd.pixelspecular_bounce = false;
    prd.pixelroughness = 1;
    prd.pixelemissive = true;
  }

  extern "C" __global__ void __miss__shadow()
  {
    // we didn't hit anything, so the light is visible
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = vec3f(1.f);
  }

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;
    
    PRD prd;
    prd.random.init(ix+optixLaunchParams.frame.size.x*iy,
                    optixLaunchParams.frame.frameID);
    prd.pixelColor = vec3f(0.f);
    prd.recursion = 1;
    prd.contribution_to_pixel = 1;
    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    int numPixelSamples = optixLaunchParams.numPixelSamples;

    vec3f pixelColor = 0.f;
    vec3f pixelNormal = 0.f;
    vec3f pixelAlbedo = 0.f;

    vec3f pixelScolor = 0.f;
    vec3f pixelDcolor = 0.f;
    float pixeldepth = 0;
    float pixelroughness = 0.f;
    bool  pixelspecular_bounce = false;
    bool  pixelmetallic = false;
    bool  pixelemissive = false;

    // printf("Generating samples at pixel %d %d\n", ix, iy);
    for (int sampleID=0;sampleID<numPixelSamples;sampleID++) {
      // normalized screen plane position, in [0,1]^2

      // iw: note for denoising that's not actually correct - if we
      // assume that the camera should only(!) cover the denoised
      // screen then the actual screen plane we shuld be using during
      // rendreing is slightly larger than [0,1]^2
      vec2f screen(vec2f(ix+prd.random(),iy+prd.random())
                   / vec2f(optixLaunchParams.frame.size));
      // screen
      //   = screen
      //   * vec2f(optixLaunchParams.frame.denoisedSize)
      //   * vec2f(optixLaunchParams.frame.size)
      //   - 0.5f*(vec2f(optixLaunchParams.frame.size)
      //           -
      //           vec2f(optixLaunchParams.frame.denoisedSize)
      //           );
      if (ix == 500 && iy == 500) {
          if (sampleID % 10 == 0) {
              // printf("Generating sample %d, at pixel %d %d\n", sampleID, ix, iy);
          }
      }
      // generate ray direction
      vec3f rayDir = normalize(camera.direction
                               + (screen.x - 0.5f) * camera.horizontal
                               + (screen.y - 0.5f) * camera.vertical);
      prd.pixelColor = vec3f(0.f);
      prd.recursion = 1;
      prd.contribution_to_pixel = 1;
      optixTrace(optixLaunchParams.traversable,
                 camera.position,
                 rayDir,
                 0.f,    // tmin
                 1e20f,  // tmax
                 0.0f,   // rayTime
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                 RADIANCE_RAY_TYPE,            // SBT offset
                 RAY_TYPE_COUNT,               // SBT stride
                 RADIANCE_RAY_TYPE,            // missSBTIndex 
                 u0, u1 );
      pixelColor  += prd.pixelColor;
      pixelNormal += prd.pixelNormal;
      pixelAlbedo += prd.pixelAlbedo;

      pixelScolor += prd.pixelScolor;
      pixelDcolor += prd.pixelDcolor;
      pixeldepth += prd.pixeldepth;
      pixelroughness += prd.pixelroughness;
      pixelspecular_bounce = prd.pixelspecular_bounce;
      pixelmetallic = prd.pixelmetallic;
      pixelemissive = prd.pixelemissive;

    }
    pixelNormal = pixelNormal / numPixelSamples;
    // move to camera space
    pixelNormal = normalize(WorldToLocal(normalize(pixelNormal), normalize(camera.direction), normalize(camera.horizontal)));

    vec4f rgba(pixelColor/numPixelSamples,1.f);
    vec4f albedo(pixelAlbedo/numPixelSamples,1.f);
    vec4f normal(pixelNormal/2 +vec3f( 0.5),1.f);
    vec4f diffuse(pixelDcolor / numPixelSamples, 1.f);
    vec4f specular(pixelScolor / numPixelSamples, 1.f);
    
    float depth = pixeldepth / (float)numPixelSamples;
    float roughness = pixelroughness / (float)numPixelSamples;


    // and write/accumulate to frame buffer ...
    // bconst uint32_t 
    uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

    // When not doing production, do accumulation
    if (!optixLaunchParams.production && optixLaunchParams.frame.frameID > 0) {
      rgba
        += float(optixLaunchParams.frame.frameID)
        *  vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
      rgba /= (optixLaunchParams.frame.frameID+1.f);
    }

    optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;
    optixLaunchParams.frame.albedoBuffer[fbIndex] = (float4)albedo;
    optixLaunchParams.frame.normalBuffer[fbIndex] = (float4)normal;
    optixLaunchParams.frame.DcolorBuffer[fbIndex] = (float4)diffuse;
    optixLaunchParams.frame.ScolorBuffer[fbIndex] = (float4)specular;
    optixLaunchParams.frame.depth[fbIndex] = depth;
    optixLaunchParams.frame.roughness[fbIndex] = roughness;
    optixLaunchParams.frame.specular_bounce[fbIndex] = pixelspecular_bounce;
    optixLaunchParams.frame.metallic[fbIndex] = pixelmetallic;
    optixLaunchParams.frame.emissive[fbIndex] = pixelemissive;
  }
  
} // ::osc
