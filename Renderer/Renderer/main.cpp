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

#include "SampleRenderer.h"
#include "Scene.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <filesystem>
#include <random>

inline float random(std::mt19937& rnd) {
    std::uniform_real_distribution<> dist(0, 1);
    return dist(rnd);
}

inline int number_of_files_in_directory(const std::string& path) {
    int count = 0;
    std::filesystem::path p1(path);
    for (auto& p : std::filesystem::directory_iterator(p1)) ++count;
    return count;

}
/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct SampleWindow : public GLFCameraWindow
  {
    SampleWindow(const std::string &title,
                 const Scene *scene,
                 const Camera &camera,
                 const float worldScale)
      : GLFCameraWindow(title,camera.from,camera.at,camera.up,worldScale),
        sample(scene)
    {
      sample.setCamera(camera);
    }
    
    virtual void render() override
    {
      if (cameraFrame.modified) {
        sample.setCamera(Camera{ cameraFrame.get_from(),
                                 cameraFrame.get_at(),
                                 cameraFrame.get_up() });
        cameraFrame.modified = false;
      }
      sample.render();
    }
    
    virtual void draw() override
    {
      sample.downloadPixels(pixels.data());

      if (fbTexture == 0)
        glGenTextures(1, &fbTexture);
      
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      GLenum texFormat = GL_RGBA;
      GLenum texelType = GL_UNSIGNED_BYTE;
      glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                   texelType, pixels.data());

      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
      
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();
    }
    
    virtual void resize(const vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x*newSize.y);
    }

    virtual void key(int key, int mods)
    {
      if (key == 'D' || key == ' ' || key == 'd') {
        sample.denoiserOn = !sample.denoiserOn;
        std::cout << "denoising now " << (sample.denoiserOn?"ON":"OFF") << std::endl;
      }
      if (key == 'A' || key == 'a') {
        sample.accumulate = !sample.accumulate;
        std::cout << "accumulation/progressive refinement now " << (sample.accumulate?"ON":"OFF") << std::endl;
      }
      if (key == ',') {
        sample.launchParams.numPixelSamples
          = std::max(1,sample.launchParams.numPixelSamples-1);
        std::cout << "num samples/pixel now "
                  << sample.launchParams.numPixelSamples << std::endl;
      }
      if (key == '.') {
        sample.launchParams.numPixelSamples
          = std::max(1,sample.launchParams.numPixelSamples+1);
        std::cout << "num samples/pixel now "
                  << sample.launchParams.numPixelSamples << std::endl;
      }
    }
    

    vec2i                 fbSize;
    GLuint                fbTexture {0};
    SampleRenderer        sample;
    std::vector<uint32_t> pixels;
  };
  
  class imageCreater
  {
  public:
      imageCreater(Scene* Iscene, Camera& camera) : sample(Iscene)
      {
          row = 1024;
          column = 1024;
          totalsize = row * column;
          sample.setCamera(camera);
          sample.resize(vec2i(row, column));
          frame.albedoBuffer.resize(totalsize);
          frame.Scolor.resize(totalsize);
          frame.Dcolor.resize(totalsize);
          frame.color.resize(totalsize);
          frame.depth.resize(totalsize);
          frame.roughness.resize(totalsize);
          frame.normalBuffer.resize(totalsize);
          frame.metallic = new bool[totalsize];
          frame.emissive = new bool[totalsize];
          frame.specular_bounce = new bool[totalsize];
          scene = Iscene;
      }
      ~imageCreater() {}

      void render(const std::string& outputPath) {
          int samples = number_of_files_in_directory(outputPath);
          std::string newPath = outputPath + "/" + std::to_string(samples);
          std::filesystem::create_directory(std::filesystem::path(newPath));

          // first render 8 single sampled images
          for (int i = 0; i < 8; i++) {
              sample.productionRender(i, 1, false);
              sample.downloadframe(frame);
              writeOutput(i, newPath);
          }
          sample.productionRender(1024, 768, true);
          sample.downloadframe(frame);

          std::string finalImagePath = std::string(newPath) + std::string("/") + std::string("_reference.hdr");
          stbi_write_hdr(finalImagePath.c_str(), row, column, 4, (float*)frame.color.data());
      }
      
      void randomizeCamera(std::mt19937& rnd){

        if (random(rnd) > 0.5) {
            vec3f lower = vec3f(-2.5, 2.79243, -0.18591);
            vec3f diff = vec3f(-0.5, 0, -2.5) - lower;
            vec3f camera_pos = vec3f((random(rnd) / 3 + 0.33) * diff.x, (random(rnd) / 3 + 0.33) * diff.y, (random(rnd) / 3 + 0.33) * diff.z) + lower;

            lower = vec3f(0.5, 2.79243, -0.18591);
            diff = vec3f(2.5, 0, -2.5) - lower;
            vec3f camera_lookat = vec3f((random(rnd) / 3 + 0.33) * diff.x, (random(rnd) / 3 + 0.33) * diff.y, (random(rnd) / 3 + 0.33) * diff.z) + lower;
            Camera camera = { camera_pos,camera_lookat,vec3f(0.f,1.f,0.f) };
            sample.setCamera(camera);
        }
        else {
            vec3f lower = vec3f(0.5, 2.79243, -0.18591);
            vec3f diff = vec3f(2.5, 0, -2.5) - lower;
            
            vec3f camera_pos = vec3f((random(rnd) / 3 + 0.33) * diff.x, (random(rnd) / 3 + 0.33) * diff.y, (random(rnd) / 3 + 0.33) * diff.z) + lower;

            lower = vec3f(-2.5, 2.79243, -0.18591);
            diff = vec3f(-0.5, 0, -2.5) - lower;
            vec3f camera_lookat = vec3f((random(rnd) / 3 + 0.33) * diff.x, (random(rnd) / 3 + 0.33) * diff.y, (random(rnd) / 3 + 0.33) * diff.z) + lower;
            Camera camera = { camera_pos,camera_lookat,vec3f(0.f,1.f,0.f) };
            sample.setCamera(camera);
        }
      }
      void build() {
          sample.buildScene();
      }
      fullframe frame;
   private:
      
      void writeOutput(int sampleID, const std::string& path) {
          std::cout << "outputing image" << std::endl;
          std::string finalImagePath = std::string(path) + std::string("/") + std::to_string(sampleID) + std::string("_finalImage.hdr");
          stbi_write_hdr(finalImagePath.c_str(), row, column, 4, (float*)frame.color.data());

          finalImagePath = std::string(path) + std::string("/") + std::to_string(sampleID) + std::string("_diffuse.hdr");
          stbi_write_hdr(finalImagePath.c_str(), row, column, 4, (float*)frame.Dcolor.data());

          finalImagePath = std::string(path) + std::string("/") + std::to_string(sampleID) + std::string("_specular.hdr");
          stbi_write_hdr(finalImagePath.c_str(), row, column, 4, (float*)frame.Scolor.data());

          finalImagePath = std::string(path) + std::string("/") + std::to_string(sampleID) + std::string("_albedo.hdr");
          stbi_write_hdr(finalImagePath.c_str(), row, column, 4, (float*)frame.albedoBuffer.data());

          finalImagePath = std::string(path) + std::string("/") + std::to_string(sampleID) + std::string("_roughness.hdr");
          stbi_write_hdr(finalImagePath.c_str(), row, column, 1, (float*)frame.roughness.data());
          
          float maxD = 0;
          for (auto depth : frame.depth) {
              maxD = fmax(maxD, depth);
          }
          for (int i = 0; i < frame.depth.size(); i++) {
              frame.depth[i] = frame.depth[i] > 0 ? frame.depth[i] / maxD : frame.depth[i];
          }
          finalImagePath = std::string(path) + std::string("/") + std::to_string(sampleID) + std::string("_depth.hdr");
          stbi_write_hdr(finalImagePath.c_str(), row, column, 1, (float*)frame.depth.data());


          finalImagePath = std::string(path) + std::string("/") + std::to_string(sampleID) + std::string("_normal.hdr");
          stbi_write_hdr(finalImagePath.c_str(), row, column, 4, (float*)frame.normalBuffer.data());

          finalImagePath = std::string(path) + std::string("/") + std::to_string(sampleID) + std::string("_metallic.png");
          std::vector<UINT8> metallic;
          for (int rowID = 0; rowID < row * column; rowID++) {
              metallic.push_back(frame.metallic[rowID] ? 255 : 0);
          }
          stbi_write_png(finalImagePath.c_str(), row, column, 1, metallic.data(), row);

          finalImagePath = std::string(path) + std::string("/") + std::to_string(sampleID) + std::string("_specularReflect.png");
          std::vector<UINT8> specularReflect;
          for (int rowID = 0; rowID < row * column; rowID++) {
              specularReflect.push_back(frame.specular_bounce[rowID] ? 255 : 0);
          }
          stbi_write_png(finalImagePath.c_str(), row, column, 1, specularReflect.data(), row);

          finalImagePath = std::string(path) + std::string("/") + std::to_string(sampleID) + std::string("_emissive.png");
          std::vector<UINT8> emissive;
          for (int rowID = 0; rowID < row * column; rowID++) {
              emissive.push_back(frame.emissive[rowID] ? 255 : 0);
          }
          stbi_write_png(finalImagePath.c_str(), row, column, 1, emissive.data(), row);
      }
      int row, column, totalsize;
      SampleRenderer sample;
      Scene* scene;
  };


  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    
    std::random_device rd;   // non-deterministic generator
    std::mt19937 rnd(rd());

    Scene* testScene = new Scene;
    
    testScene->loadBaseScene("C:/Users/Alsto/Desktop/bathroom/scene.pbrt");
    std::cout << "Base Scene Loaded" << std::endl;
    // testScene->loadAdditionalScene("C:/Users/Alsto/OneDrive - UC San Diego/CSE 274/models", 20, rnd);
    std::cout << "additional mesh Loaded" << std::endl;
    Camera camera = { /*from*/vec3f(0.158333f, 0.991688f, -0.235041f),
        /* at */vec3f(-2.89823, 1, -3.04425),
        /* up */vec3f(0.f,1.f,0.f) };
    testScene->bounds.lower = vec3f(-2.5, 2.79243, -0.18591);
    testScene->bounds.upper = vec3f(2.5, 0, -2.5);

    // testScene->randomizeOBJs("C:/Users/Alsto/OneDrive - UC San Diego/CSE 274/textures", rnd);
    testScene->addRandomizeLight(rnd);
    std::cout << "obj randomized" << std::endl;
    if (ac > 1 ) {
        try {

            const float worldScale = length(testScene->bounds.span());

            SampleWindow* window = new SampleWindow("Optix 7 Course Example",
                testScene, camera, worldScale);
            window->enableFlyMode();

            std::cout << "Press 'a' to enable/disable accumulation/progressive refinement" << std::endl;
            std::cout << "Press ' ' to enable/disable denoising" << std::endl;
            std::cout << "Press ',' to reduce the number of paths/pixel" << std::endl;
            std::cout << "Press '.' to increase the number of paths/pixel" << std::endl;
            window->run();

        }
        catch (std::runtime_error& e) {
            std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
            std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
            exit(1);
        }
    }
    else {
        imageCreater crtr(testScene, camera);
        crtr.randomizeCamera(rnd);
        crtr.build();
        std::cout << "Start rendering" << std::endl;
        crtr.render("C:/Users/Alsto/OneDrive - UC San Diego/CSE 274/dataset");

    }
    return 0;
  }
  
} // ::osc
