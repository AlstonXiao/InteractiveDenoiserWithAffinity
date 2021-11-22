#pragma once
#include<vector>
#include<map>
#include"Model.h"

namespace osc {
	using namespace gdt;
	

	class Scene {

	public:
		std::vector<Material*> material_list;

		std::map<std::string, int> knownTextures;
		std::vector<Texture* > texture_list;

		Model* base_model;
		std::vector<Model* > additional_model_list;
		std::vector<QuadLight > light_list;

		~Scene()
		{
			delete base_model;
			for (auto mesh : additional_model_list) delete mesh;
			for (auto texture : texture_list) delete texture;
			for (auto material : material_list) delete material;
		}
		
		void loadBaseScene(const std::string& pbrtFile);
		void loadAdditionalScene(const std::string& modelPath, int numberOfObjects, std::mt19937& rnd);
		void addMesh(const std::string& objFile);
		void randomizeOBJs(const std::string& texturePath, std::mt19937& rnd);
		void addRandomizeLight(std::mt19937& rnd);
		box3f bounds;
	};

	
	
	


}