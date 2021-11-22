#pragma once

#include "gdt/math/AffineSpace.h"
#include "3rdParty/stb_image.h"
#include <vector>
#include <map>
#include <random>
namespace osc {
	using namespace gdt;
	enum class MatType { matte, plastic, substrate, metal, uber, mirror };
	class Material {

	public:
		const float CopperN[56] = {
		1.400313, 1.38,  1.358438, 1.34,  1.329063, 1.325, 1.3325,   1.34,
		1.334375, 1.325, 1.317812, 1.31,  1.300313, 1.29,  1.281563, 1.27,
		1.249062, 1.225, 1.2,      1.18,  1.174375, 1.175, 1.1775,   1.18,
		1.178125, 1.175, 1.172812, 1.17,  1.165312, 1.16,  1.155312, 1.15,
		1.142812, 1.135, 1.131562, 1.12,  1.092437, 1.04,  0.950375, 0.826,
		0.645875, 0.468, 0.35125,  0.272, 0.230813, 0.214, 0.20925,  0.213,
		0.21625,  0.223, 0.2365,   0.25,  0.254188, 0.26,  0.28,     0.3 };

		const float CopperK[56] = {
		1.662125, 1.687, 1.703313, 1.72,  1.744563, 1.77,  1.791625, 1.81,
		1.822125, 1.834, 1.85175,  1.872, 1.89425,  1.916, 1.931688, 1.95,
		1.972438, 2.015, 2.121562, 2.21,  2.177188, 2.13,  2.160063, 2.21,
		2.249938, 2.289, 2.326,    2.362, 2.397625, 2.433, 2.469187, 2.504,
		2.535875, 2.564, 2.589625, 2.605, 2.595562, 2.583, 2.5765,   2.599,
		2.678062, 2.809, 3.01075,  3.24,  3.458187, 3.67,  3.863125, 4.05,
		4.239563, 4.43,  4.619563, 4.817, 5.034125, 5.26,  5.485625, 5.717 };

		vec3f kd;
		vec3f ks;
		float roughness_square;
		int kd_map_id;
		int ks_map_id;
		bool has_kd_map;
		bool has_ks_map;
		MatType type;

		void randomize(int textureCount, std::mt19937& rnd);
		Material(MatType type) : type(type) {
			kd = vec3f(0.25, 0.25, 0.25);
			ks = vec3f(0.25, 0.25, 0.25);
			roughness_square = 0.01;
			kd_map_id = -1;
			ks_map_id = -1;
			has_kd_map = false;
			has_ks_map = false;
		}
	};


	class matteMaterial : public Material {
	public:
		matteMaterial() : Material(MatType::matte) {
			kd = vec3f(0.5, 0.5, 0.5);
			roughness_square = 0;
		}
	};

	class plasticMaterial : public Material {
	public:
		plasticMaterial() : Material(MatType::plastic) {
		}
	};

	class substrateMaterial : public Material {
	public:
		substrateMaterial() : Material(MatType::substrate) {
			kd = vec3f(0.5, 0.5, 0.5);
			ks = vec3f(0.5, 0.5, 0.5);
		}
	};

	class metalMaterial : public Material {
	public:
		metalMaterial() : Material(MatType::metal) {
			// kd for eta
			kd = vec3f(0.19999069, 0.92208463, 1.09987593);
			// ks for k
			ks = vec3f(3.90463543, 2.44763327, 2.13765264);
			roughness_square = 0.0001;
		}
	};

	class uberMaterial : public Material {
	public:
		uberMaterial() : Material(MatType::uber) {
		}
	};

	class mirrorMaterial : public Material {
	public:
		mirrorMaterial() : Material(MatType::mirror) {
			kd = vec3f(0, 0, 0);
			ks = vec3f(0.9, 0.9, 0.9);
		}
	};

	struct Texture {
		~Texture()
		{
			if (pixel) delete[] pixel;
		}

		uint32_t* pixel{ nullptr };
		vec2i     resolution{ -1 };
	};

	/*! load a texture (if not already loaded), and return its ID in the
	  model's textures[] vector. Textures that could not get loaded
	  return -1 */
	int loadTexture(std::vector<Texture* >& texture_list,
		std::map<std::string, int>& knownTextures,
		const std::string& inFileName,
		const std::string& modelPath);
}