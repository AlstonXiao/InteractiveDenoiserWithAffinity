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

#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"

#define TINYPLY_IMPLEMENTATION
#include "3rdParty/tinyply.h"

//std
#include <set>

namespace std {
  inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
  {
    if (a.vertex_index < b.vertex_index) return true;
    if (a.vertex_index > b.vertex_index) return false;
    
    if (a.normal_index < b.normal_index) return true;
    if (a.normal_index > b.normal_index) return false;
    
    if (a.texcoord_index < b.texcoord_index) return true;
    if (a.texcoord_index > b.texcoord_index) return false;
    
    return false;
  }
}

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
   using namespace std;
  /*! find vertex with given position, normal, texcoord, and return
      its vertex ID, or, if it doesn't exit, add it to the mesh, and
      its just-created index */
  int addVertex(TriangleMesh *mesh,
                tinyobj::attrib_t &attributes,
                const tinyobj::index_t &idx,
                std::map<tinyobj::index_t,int> &knownVertices)
  {
    if (knownVertices.find(idx) != knownVertices.end())
      return knownVertices[idx];

    const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
    const vec3f *normal_array   = (const vec3f*)attributes.normals.data();
    const vec2f *texcoord_array = (const vec2f*)attributes.texcoords.data();
    
    int newID = (int)mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
      while (mesh->normal.size() < mesh->vertex.size())
        mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
      while (mesh->texcoord.size() < mesh->vertex.size())
        mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    // just for sanity's sake:
    if (mesh->texcoord.size() > 0)
      mesh->texcoord.resize(mesh->vertex.size());
    // just for sanity's sake:
    if (mesh->normal.size() > 0)
      mesh->normal.resize(mesh->vertex.size());
    
    return newID;
  }
  
  Model *loadOBJ(const std::string &objFile, std::vector<Texture* > &texture_list, std::map<std::string, int> &knownTextures)
  {
    Model *model = new Model;

    const std::string modelDir
      = objFile.substr(0,objFile.rfind('/')+1);
    
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK
      = tinyobj::LoadObj(&attributes,
                         &shapes,
                         &materials,
                         &err,
						 &err,
                         objFile.c_str(),
                         modelDir.c_str(),
                         /* triangulate */true);
    if (!readOK) {
      throw std::runtime_error("Could not read OBJ model from "+objFile+" : "+err);
    }

    if (materials.empty())
      throw std::runtime_error("could not parse materials ...");

    // std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;

    for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
      tinyobj::shape_t &shape = shapes[shapeID];

      std::set<int> materialIDs;
      for (auto faceMatID : shape.mesh.material_ids)
        materialIDs.insert(faceMatID);
      
      std::map<tinyobj::index_t,int> knownVertices;
      
      for (int materialID : materialIDs) {
        TriangleMesh *mesh = new TriangleMesh;
        
        for (int faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
          if (shape.mesh.material_ids[faceID] != materialID) continue;
          tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
          tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
          tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
          
          vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                    addVertex(mesh, attributes, idx1, knownVertices),
                    addVertex(mesh, attributes, idx2, knownVertices));
          mesh->index.push_back(idx);
          mesh->mat = new uberMaterial();

          if (materialID >= 0) {
              mesh->mat->kd = (const vec3f&)materials[materialID].diffuse;
              mesh->mat->ks = (const vec3f&)materials[materialID].specular;
              mesh->mat->roughness_square = pow((materials[materialID].shininess == 0) ? 0. : (1.f / materials[materialID].shininess), 2);

              mesh->mat->kd_map_id = loadTexture(texture_list,
                  knownTextures,
                  materials[materialID].diffuse_texname,
                  modelDir);

              mesh->mat->ks_map_id = loadTexture(texture_list,
                  knownTextures,
                  materials[materialID].specular_texname,
                  modelDir);
          }
        }

        if (mesh->vertex.empty())
          delete mesh;
        else
          model->meshes.push_back(mesh);
      }
    }

    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes)
      for (auto vtx : mesh->vertex)
        model->bounds.extend(vtx);
    
    // std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
    return model;
  }

  void tokenize(std::string const& str, const char delim,
      std::vector<std::string>& out)
  {
      size_t start;
      size_t end = 0;

      while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
      {
          end = str.find(delim, start);
          out.push_back(str.substr(start, end - start));
      }
  }

  inline void removeQuote(string& str) {
      str.erase(std::remove(str.begin(), str.end(), '\"'), str.end());
  }

  void processMaterial(Material* mat, std::vector<std::string>& commands, std::map<std::string, int> textureMap) {
      for (size_t i = 7; i < commands.size() - 1; ) {
          string type = (commands[i]);
          removeQuote(type);
          string entry = (commands[i + 1]);
          removeQuote(entry);
          if (type == "rgb" && (entry == "Ks"|| entry == "k")) {
              mat->ks = vec3f(stof(commands[i + 3]), stof(commands[i + 4]), stof(commands[i + 5]));
              i = i + 7;
          }
          else if (type == "rgb" && (entry == "Kd" || entry == "eta")) {
              mat->kd = vec3f(stof(commands[i + 3]), stof(commands[i + 4]), stof(commands[i + 5]));
              i = i + 7;
          }
          else if (type == "texture" && entry == "Ks") {
              string textureName = commands[i + 3];
              removeQuote(textureName);
              mat->has_ks_map = true;
              mat->ks_map_id = textureMap[textureName];
              i = i + 5;
          }
          else if (type == "texture" && entry == "Kd") {
              string textureName = commands[i + 3];
              removeQuote(textureName);
              mat->has_kd_map = true;
              mat->kd_map_id = textureMap[textureName];
              i = i + 5;
          }
          else if (type == "float" && entry == "uroughness") {
              mat->roughness_square = stof(commands[i + 3]);
              i = i + 5;
          }
          else if (type == "float" && entry == "vroughness") {
              mat->roughness_square *= stof(commands[i + 3]);
              i = i + 5;
          }
          else if (type == "float" && entry == "roughness") {
              mat->roughness_square = stof(commands[i + 3]) * stof(commands[i + 3]);
              i = i + 5;
          }
          else {
              i = i + 1;
          }
      }
  }

  TriangleMesh* loadPly(const std::string& plyfile) {
      //std::cout << "........................................................................\n";
      //std::cout << "Now Reading: " << plyfile << std::endl;

      std::unique_ptr<std::istream> file_stream;
      std::vector<uint8_t> byte_buffer;
      using namespace tinyply;
      try {
            // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a 
            // stream is a net win for parsing speed, about 40% faster. 

            file_stream.reset(new std::ifstream(plyfile, std::ios::binary));
            if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + plyfile);

            file_stream->seekg(0, std::ios::end);
            const float size_mb = file_stream->tellg() * float(1e-6);
            file_stream->seekg(0, std::ios::beg);

            PlyFile file;
            file.parse_header(*file_stream);

            // std::cout << "\t[ply_header] Type: " << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
            //for (const auto& c : file.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
            //for (const auto& c : file.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl;

            //for (const auto& e : file.get_elements())
            //{
            //    std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
            //    for (const auto& p : e.properties)
            //    {
            //        std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
            //        if (p.isList) std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
            //        std::cout << std::endl;
            //    }
            //}

            // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers. 
            // See examples below on how to marry your own application-specific data structures with this one. 
            std::shared_ptr<PlyData> vertices, normals, texcoords, faces;

            // The header information can be used to programmatically extract properties on elements
            // known to exist in the header prior to reading the data. For brevity of this sample, properties 
            // like vertex position are hard-coded: 
            try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
            catch (const std::exception& e) {} // std::cerr << "tinyply exception: " << e.what() << std::endl; }

            try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
            catch (const std::exception& e) {} // std::cerr << "tinyply exception: " << e.what() << std::endl; }

            try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); }
            catch (const std::exception& e) {} // std::cerr << "tinyply exception: " << e.what() << std::endl; }

            // Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
            // arbitrary ply files, it is best to leave this 0. 
            try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
            catch (const std::exception& e) {} // std::cerr << "tinyply exception: " << e.what() << std::endl; }

            file.read(*file_stream);

            //if (vertices)   std::cout << "\tRead " << vertices->count << " total vertices " << std::endl;
            //if (normals)    std::cout << "\tRead " << normals->count << " total vertex normals " << std::endl;
            //if (texcoords)  std::cout << "\tRead " << texcoords->count << " total vertex texcoords " << std::endl;
            //if (faces)      std::cout << "\tRead " << faces->count << " total faces (triangles) " << std::endl;

            // Example One: converting to your own application types
            TriangleMesh* mesh = new TriangleMesh();
            
            const size_t numVerticesBytes = vertices->buffer.size_bytes();
            std::vector<vec3f> verts(vertices->count);
            std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);
            mesh->vertex = verts;

            if (normals) {
                const size_t numNormalBytes = normals->buffer.size_bytes();
                std::vector<vec3f> norms(normals->count);
                std::memcpy(norms.data(), normals->buffer.get(), numNormalBytes);
                mesh->normal = norms;
            } 

            if (texcoords) {
                const size_t numTextBytes = texcoords->buffer.size_bytes();
                std::vector<vec2f> texts(texcoords->count);
                std::memcpy(texts.data(), texcoords->buffer.get(), numTextBytes);
                mesh->texcoord = texts;
            }

            const size_t numFaceBytes = faces->buffer.size_bytes();
            std::vector<vec3i> facess(faces->count);
            std::memcpy(facess.data(), faces->buffer.get(), numFaceBytes);
            
            mesh->index = facess;
            return mesh;
            // Example Two: converting to your own application type
            //{
            //    std::vector<float3> verts_floats;
            //    std::vector<double3> verts_doubles;
            //    if (vertices->t == tinyply::Type::FLOAT32) { /* as floats ... */ }
            //    if (vertices->t == tinyply::Type::FLOAT64) { /* as doubles ... */ }
            //}
        }
        catch (const std::exception& e)
        {
            std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
        }
      
  }

  Model* loadPBRT(const std::string& pbrtFile, std::vector<Texture* >& texture_list, std::map<std::string, int>& knownTextures, std::vector<QuadLight>& lightList) {
      Model* model = new Model;
      std::map<std::string, Material *> materialMap;
      std::map<std::string, int> textureMap;

      const std::string modelDir
          = pbrtFile.substr(0, pbrtFile.rfind('/') );

      std::ifstream infile(pbrtFile);

      std::string line;
      Material* currentMat = new uberMaterial();
      bool skip = false;
      while (std::getline(infile, line)) {
          std::vector<std::string> commands;
          line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
          tokenize(line, ' ', commands);
          if (commands.size() == 0) continue;
          if (commands[0] == "AttributeBegin") {
              skip = true;
              string areaLine;
              std::getline(infile, areaLine);
              std::vector<std::string> areaLight;
              areaLine.erase(std::remove(areaLine.begin(), areaLine.end(), '\t'), areaLine.end());
              tokenize(areaLine, ' ', areaLight);

              QuadLight light;
              light.power = vec3f(stof(areaLight[5]), stof(areaLight[6]), stof(areaLight[7]));
              std::getline(infile, areaLine);
              std::getline(infile, areaLine);
              std::vector<std::string> areaLightDetail;
              areaLine.erase(std::remove(areaLine.begin(), areaLine.end(), '\t'), areaLine.end());
              tokenize(areaLine, ' ', areaLightDetail);
              vec3f origin = vec3f(stof(areaLightDetail[15]), stof(areaLightDetail[16]), stof(areaLightDetail[17]));
              vec3f leftEdge = vec3f(stof(areaLightDetail[18]), stof(areaLightDetail[19]), stof(areaLightDetail[20]));
              vec3f rightEdge = vec3f(stof(areaLightDetail[24]), stof(areaLightDetail[25]), stof(areaLightDetail[26]));
              
              light.origin = origin;
              light.du = leftEdge - origin;
              light.dv = rightEdge - origin;
              lightList.push_back(light);
          }
          if (commands[0] == "AttributeEnd") {
              skip = false;
          }
          if (skip) continue;
          if (commands[0] == "Texture") {
              string type = (commands[2]);
              removeQuote(type);
              if (type != "spectrum") continue;

              string name = commands[1];
              removeQuote(name);

              string file = commands[7];
              removeQuote(file);
              string path = modelDir + file;
              int texture_id = loadTexture(texture_list, knownTextures, file, modelDir);
              textureMap[name] = texture_id;
          }
          if (commands[0] == "MakeNamedMaterial") {
              string name = commands[1];
              removeQuote(name);
              string type = (commands[5]);
              removeQuote(type);
               if (type == "substrate") {
                  substrateMaterial* mat = new substrateMaterial();
                  processMaterial(mat, commands, textureMap);
                  materialMap[name] = mat;
              }
              else if (type == "matte") {
                  matteMaterial* mat = new matteMaterial();
                  processMaterial(mat, commands, textureMap);
                  materialMap[name] = mat;
              }
              else if (type == "metal") {
                   metalMaterial* mat = new metalMaterial();
                  processMaterial(mat, commands, textureMap);
                  materialMap[name] = mat;
              }
              else if (type == "mirror") {
                  mirrorMaterial* mat = new mirrorMaterial();
                  processMaterial(mat, commands, textureMap);
                  materialMap[name] = mat;
              } else {
                  uberMaterial* mat = new uberMaterial();
                  processMaterial(mat, commands, textureMap);
                  materialMap[name] = mat;
              }
          }
          if (commands[0] == "NamedMaterial") {
              removeQuote(commands[1]);
              currentMat = materialMap[commands[1]];
          }
          if (commands[0] == "Shape") {
              string type = (commands[1]);
              removeQuote(type);
              if (type == "plymesh") {
                  string file = commands[5];
                  removeQuote(file);
                  string path = modelDir + "/" + file;
                  TriangleMesh* mesh = loadPly(path);
                  mesh->mat = currentMat;
                  model->meshes.push_back(mesh);
              }
              else {
                  TriangleMesh* mesh = new TriangleMesh;
                  mesh->mat = currentMat;
                  size_t i = 5;
                  while (commands[i] != "]") {
                      mesh->index.push_back(vec3i(stoi(commands[i]), stoi(commands[i + 1]), stoi(commands[i + 2])));
                      i = i + 3;
                  }
                  i = i + 4;
                  while (commands[i] != "]") {
                      mesh->vertex.push_back(vec3f(stof(commands[i]), stof(commands[i + 1]), stof(commands[i + 2])));
                      i = i + 3;
                  }
                  i = i + 4;
                  while (commands[i] != "]") {
                      mesh->normal.push_back(vec3f(stof(commands[i]), stof(commands[i + 1]), stof(commands[i + 2])));
                      i = i + 3;
                  }
                  i = i + 4;
                  while (commands[i] != "]") {
                      mesh->texcoord.push_back(vec2f(stof(commands[i]), stof(commands[i + 1])));
                      i = i + 2;
                  }
                  model->meshes.push_back(mesh);
              }
          }
      }
      for (auto mesh : model->meshes)
          for (auto vtx : mesh->vertex)
              model->bounds.extend(vtx);
      std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
      return model;
  }
}
