#include "Material.h"
#include <random>

namespace osc {
	inline float random(std::mt19937& rnd) {
		std::uniform_real_distribution<> dist(0, 1);
		return dist(rnd);
	}
	int loadTexture(std::vector<Texture* >& texture_list,
		std::map<std::string, int>& knownTextures,
		const std::string& inFileName,
		const std::string& modelPath)
	{
		if (inFileName == "")
			return -1;

		if (knownTextures.find(inFileName) != knownTextures.end())
			return knownTextures[inFileName];

		std::string fileName = inFileName;
		// first, fix backspaces:
		for (auto& c : fileName)
			if (c == '\\') c = '/';
		fileName = modelPath + "/" + fileName;

		vec2i res;
		int   comp;
		unsigned char* image = stbi_load(fileName.c_str(),
			&res.x, &res.y, &comp, STBI_rgb_alpha);
		int textureID = -1;
		if (image) {
			textureID = (int)texture_list.size();
			Texture* texture = new Texture;
			texture->resolution = res;
			texture->pixel = (uint32_t*)image;

			/* iw - actually, it seems that stbi loads the pictures
			   mirrored along the y axis - mirror them here */
			for (int y = 0; y < res.y / 2; y++) {
				uint32_t* line_y = texture->pixel + y * res.x;
				uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
				int mirror_y = res.y - 1 - y;
				for (int x = 0; x < res.x; x++) {
					std::swap(line_y[x], mirrored_y[x]);
				}
			}

			texture_list.push_back(texture);
		}
		else {
			std::cout << GDT_TERMINAL_RED
				<< "Could not load texture from " << fileName << "!"
				<< GDT_TERMINAL_DEFAULT << std::endl;
		}

		knownTextures[inFileName] = textureID;
		return textureID;
	}
	
	void Material::randomize(int textureCount, std::mt19937& rnd) {
		std::uniform_int_distribution<> textureDist(0, textureCount - 1);
		std::uniform_int_distribution<> matTypeDist(0, 5);
		type = MatType(matTypeDist(rnd));
		if (type == MatType::mirror && random(rnd) > 0.05) {
			type = MatType::uber;
		}
		if (type == MatType::mirror) return;

		roughness_square = pow(random(rnd), 2);
		std::uniform_int_distribution<> copperDist(0, 55);
		if (type == MatType::metal) {
			kd = vec3f(CopperN[copperDist(rnd)], CopperN[copperDist(rnd)], CopperN[copperDist(rnd)]);
			ks = vec3f(CopperK[copperDist(rnd)], CopperK[copperDist(rnd)], CopperK[copperDist(rnd)]);
		}
		else {
			if (has_kd_map || has_ks_map) {
				if (random(rnd) > 0.1) {
					kd_map_id = random(rnd) > 0.7 ? kd_map_id : textureDist(rnd);
				}
				else {
					has_kd_map = false;
					kd = vec3f(random(rnd), random(rnd), random(rnd));
				}
				if (random(rnd) > 0.1) {
					ks_map_id = random(rnd) > 0.7 ? ks_map_id : textureDist(rnd);
				}
				else {
					has_ks_map = false;
					ks = vec3f(random(rnd), random(rnd), random(rnd));
				}
			}
			else {
				if (random(rnd) > 0.9) {
					kd = vec3f(random(rnd), random(rnd), random(rnd));
					ks = vec3f(random(rnd), random(rnd), random(rnd));
				}
				else {
					kd_map_id =  textureDist(rnd);
					has_kd_map = true;
					ks_map_id = textureDist(rnd);
					has_ks_map = true;
				}

			}
		}
	}
}