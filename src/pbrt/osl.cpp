#include <vector>
#include <pbrt/textures.h>
#include <cstdlib>
#include <string>

// Hardware intrinsics must load before OSL
#include <immintrin.h>
#include <OSL/oslexec.h>
#include <OSL/oslquery.h>
#include <OSL/rendererservices.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/errorhandler.h>

namespace pbrt {

// ---------------------------------------------------------
// 1. OSL Renderer Services Bridge
// ---------------------------------------------------------
class PbrtRendererServices : public OSL::RendererServices {
public:
    bool get_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                    OSL::TransformationPtr xform, float time) override { return false; }
    bool get_inverse_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                            OSL::TransformationPtr xform, float time) override { return false; }
    bool get_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                    OSL::ustring from, float time) override { return false; }
    bool get_inverse_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                            OSL::ustring to, float time) override { return false; }
};

// ---------------------------------------------------------
// 2. Global OSL Engine State
// ---------------------------------------------------------
static PbrtRendererServices* oslServices = nullptr;
static OSL::ShadingSystem* shadingSystem = nullptr;
static OIIO::TextureSystem* textureSystem = nullptr;

void InitOSL() {
    if (!oslServices) {
        oslServices = new PbrtRendererServices();
        static OIIO::ErrorHandler errHandler;
        
        // Create OSL TextureSystem
        // 'true' means it will be safely shared across all rendering threads
        textureSystem = OIIO::TextureSystem::create(true);

        shadingSystem = new OSL::ShadingSystem(oslServices, textureSystem, &errHandler);
        
        // Read the OSL_SHADER_PATH environment variable to set the shader search path
        // Start with the current directory as a fallback
        std::string searchPath = ".";
        if (const char* envPath = std::getenv("OSL_SHADER_PATH")) {
        #ifdef _WIN32
            searchPath += std::string(";") + envPath;
        #else
            searchPath += std::string(":") + envPath;
        #endif
        }

        shadingSystem->attribute("searchpath:shader", searchPath);
        textureSystem->attribute("searchpath", searchPath);
        printf("[OSL] Shader and Texture search path configured as: %s\n", searchPath.c_str());
    }
}

void CleanupOSL() {
    if (shadingSystem) {
        delete shadingSystem;
        shadingSystem = nullptr;
    }
    if (textureSystem) {
        OIIO::TextureSystem::destroy(textureSystem);
        textureSystem = nullptr;
    }
    if (oslServices) {
        delete oslServices;
        oslServices = nullptr;
    }
}

// ---------------------------------------------------------
// 3. Texture Implementation
// ---------------------------------------------------------
struct OSLTextureState {
    OSL::ShaderGroupRef shaderGroup;
    const OSL::ShaderSymbol* coutSymbol = nullptr;  // Cache the symbol
};

// Float version of OSLTexture
OSLFloatTexture::OSLFloatTexture(const std::string& shaderName, const TextureParameterDictionary &parameters) {
    state = new OSLTextureState();
    if (!shadingSystem) return;

    // Turn off OSL's optimizer entirely
    // This guarantees OSL will not delete our variables behind our backs
    int opt = 0;
    shadingSystem->attribute("optimize", opt);

    state->shaderGroup = shadingSystem->ShaderGroupBegin(shaderName);

    std::string searchPath = ".";
    if (const char* envPath = std::getenv("OSL_SHADER_PATH")) {
    #ifdef _WIN32
        searchPath += std::string(";") + envPath;
    #else
        searchPath += std::string(":") + envPath;
    #endif
    }

    OSL::OSLQuery query(shaderName, searchPath);
    std::string queryErr = query.geterror();

    if (queryErr.empty()) {
        for (size_t i = 0; i < query.nparams(); ++i) {
            const OSL::OSLQuery::Parameter* param = query.getparam(i);
            
            if (!param || param->isoutput) continue; 

            std::string pName = param->name.string();

            // --- A. FLOATS ---
            if (param->type == OSL::TypeFloat) {
                float defVal = param->fdefault.empty() ? 0.0f : param->fdefault[0];
                float val = (float)parameters.GetOneFloat(pName, defVal);
                shadingSystem->Parameter(*(state->shaderGroup), param->name, OSL::TypeFloat, &val);
            }
            // --- B. INTEGERS ---
            else if (param->type == OSL::TypeInt) {
                int defVal = param->idefault.empty() ? 0 : param->idefault[0];
                int val = parameters.GetOneInt(pName, defVal);
                shadingSystem->Parameter(*(state->shaderGroup), param->name, OSL::TypeInt, &val);
            }
            // --- C. 3D TYPES (Vectors, Points, Normals, Colors) ---
            else if (param->type.basetype == OSL::TypeDesc::FLOAT && param->type.aggregate == OSL::TypeDesc::VEC3) {
                float defX = param->fdefault.size() > 0 ? param->fdefault[0] : 0.0f;
                float defY = param->fdefault.size() > 1 ? param->fdefault[1] : 0.0f;
                float defZ = param->fdefault.size() > 2 ? param->fdefault[2] : 0.0f;

                Vector3f val = parameters.GetOneVector3f(pName, Vector3f(defX, defY, defZ));
                float data[3] = { (float)val.x, (float)val.y, (float)val.z };
                shadingSystem->Parameter(*(state->shaderGroup), param->name, param->type, data);
            }
            // --- D. STRINGS ---
            else if (param->type == OSL::TypeString) {
                std::string defVal = param->sdefault.empty() ? "" : param->sdefault[0].string();
                std::string val = parameters.GetOneString(pName, defVal);
                OSL::ustring uval(val);
                shadingSystem->Parameter(*(state->shaderGroup), param->name, OSL::TypeString, &uval);
            }
        }
    } else {
        fprintf(stderr, "[OSL WARNING] OSLQuery could not read parameters for '%s'. Reason: %s\n", 
               shaderName.c_str(), queryErr.c_str());
    }

    bool shaderSuccess = shadingSystem->Shader(*(state->shaderGroup), "surface", shaderName.c_str(), "my_layer");
    if (!shaderSuccess) {
        fprintf(stderr, "\n[OSL ERROR] Failed to load '%s.oso'!\n\n", shaderName.c_str());
    }

    std::vector<OSL::ustring> outputs;
    outputs.push_back(OSL::ustring("my_layer.Cout"));
    shadingSystem->attribute(state->shaderGroup.get(), "renderer_outputs",
                             OSL::TypeDesc(OSL::TypeDesc::STRING, (int)outputs.size()), 
                             &outputs[0]);

    bool groupSuccess = shadingSystem->ShaderGroupEnd(*(state->shaderGroup));
    if (!groupSuccess) {
        fprintf(stderr, "\n[OSL ERROR] ShaderGroup linking failed! Check terminal.\n\n");
    }

    // Force JIT compilation before lookup
    // Without this, the final symbol table doesn't exist yet, and find_symbol will fail
    shadingSystem->optimize_all_groups();

    // Symbol lookup
    state->coutSymbol = shadingSystem->find_symbol(*(state->shaderGroup), OSL::ustring("my_layer.Cout"));
    if (!state->coutSymbol) {
        state->coutSymbol = shadingSystem->find_symbol(*(state->shaderGroup), OSL::ustring("Cout"));
    }

    if (!state->coutSymbol) {
        fprintf(stderr, "[OSL ERROR] Could not find output symbol in %s.\n", shaderName.c_str());
    } else {
        printf("[OSL SUCCESS] Successfully linked shader: %s\n", shaderName.c_str());
    }
}

Float OSLFloatTexture::Evaluate(TextureEvalContext ctx) const {
    if (!state || !state->shaderGroup || !shadingSystem || !state->coutSymbol) return 0.f;

    OSL::ShaderGlobals sg;
    memset(&sg, 0, sizeof(OSL::ShaderGlobals)); 
    
    // OSL math functions (like noise) require full Normal and Derivative data
    sg.P = OSL::Vec3(ctx.p.x, ctx.p.y, ctx.p.z); 
    sg.N = OSL::Vec3(ctx.n.x, ctx.n.y, ctx.n.z);
    sg.Ng = sg.N;
    sg.dPdx = OSL::Vec3(ctx.dpdx.x, ctx.dpdx.y, ctx.dpdx.z);
    sg.dPdy = OSL::Vec3(ctx.dpdy.x, ctx.dpdy.y, ctx.dpdy.z);
    sg.u = ctx.uv[0];
    sg.v = ctx.uv[1];

    sg.dudx = ctx.dudx;
    sg.dudy = ctx.dudy;
    sg.dvdx = ctx.dvdx;
    sg.dvdy = ctx.dvdy;

    thread_local OSL::PerThreadInfo* threadInfo = nullptr;
    if (!threadInfo) {
        threadInfo = shadingSystem->create_thread_info();
    }

    OSL::ShadingContext* shadingCtx = shadingSystem->get_context(threadInfo);
    
    // Capture whether execution actually succeeded
    bool success = shadingSystem->execute(*shadingCtx, *(state->shaderGroup), sg);
    float result = 0.0f;
    if (success) {
        result = *(const float*)shadingSystem->symbol_address(*shadingCtx, state->coutSymbol); 
    }

    shadingSystem->release_context(shadingCtx);
    return result;
}

std::string OSLFloatTexture::ToString() const {
    return "[ OSLFloatTexture ]";
}

// Spectrum version of OSLTexture
OSLSpectrumTexture::OSLSpectrumTexture(const std::string& shaderName, const TextureParameterDictionary &parameters, SpectrumType spectrumType) 
    : spectrumType(spectrumType) {
    state = new OSLTextureState();
    if (!shadingSystem) return;

    // Turn off OSL's optimizer entirely
    // This guarantees OSL will not delete our variables behind our backs
    int opt = 0;
    shadingSystem->attribute("optimize", opt);

    state->shaderGroup = shadingSystem->ShaderGroupBegin(shaderName);

    std::string searchPath = ".";
    if (const char* envPath = std::getenv("OSL_SHADER_PATH")) {
    #ifdef _WIN32
        searchPath += std::string(";") + envPath;
    #else
        searchPath += std::string(":") + envPath;
    #endif
    }

    OSL::OSLQuery query(shaderName, searchPath);
    std::string queryErr = query.geterror();

    if (queryErr.empty()) {
        for (size_t i = 0; i < query.nparams(); ++i) {
            const OSL::OSLQuery::Parameter* param = query.getparam(i);
            
            if (!param || param->isoutput) continue; 

            std::string pName = param->name.string();

            // --- A. FLOATS ---
            if (param->type == OSL::TypeFloat) {
                float defVal = param->fdefault.empty() ? 0.0f : param->fdefault[0];
                float val = (float)parameters.GetOneFloat(pName, defVal);
                shadingSystem->Parameter(*(state->shaderGroup), param->name, OSL::TypeFloat, &val);
            }
            // --- B. INTEGERS ---
            else if (param->type == OSL::TypeInt) {
                int defVal = param->idefault.empty() ? 0 : param->idefault[0];
                int val = parameters.GetOneInt(pName, defVal);
                shadingSystem->Parameter(*(state->shaderGroup), param->name, OSL::TypeInt, &val);
            }
            // --- C. 3D TYPES (Vectors, Points, Normals, Colors) ---
            else if (param->type.basetype == OSL::TypeDesc::FLOAT && param->type.aggregate == OSL::TypeDesc::VEC3) {
                float defX = param->fdefault.size() > 0 ? param->fdefault[0] : 0.0f;
                float defY = param->fdefault.size() > 1 ? param->fdefault[1] : 0.0f;
                float defZ = param->fdefault.size() > 2 ? param->fdefault[2] : 0.0f;

                Vector3f val = parameters.GetOneVector3f(pName, Vector3f(defX, defY, defZ));
                float data[3] = { (float)val.x, (float)val.y, (float)val.z };
                shadingSystem->Parameter(*(state->shaderGroup), param->name, param->type, data);
            }
            // --- D. STRINGS ---
            else if (param->type == OSL::TypeString) {
                std::string defVal = param->sdefault.empty() ? "" : param->sdefault[0].string();
                std::string val = parameters.GetOneString(pName, defVal);
                OSL::ustring uval(val);
                shadingSystem->Parameter(*(state->shaderGroup), param->name, OSL::TypeString, &uval);
            }
        }
    } else {
        fprintf(stderr, "[OSL WARNING] OSLQuery could not read parameters for '%s'. Reason: %s\n", 
               shaderName.c_str(), queryErr.c_str());
    }

    bool shaderSuccess = shadingSystem->Shader(*(state->shaderGroup), "surface", shaderName.c_str(), "my_layer");
    if (!shaderSuccess) {
        fprintf(stderr, "\n[OSL ERROR] Failed to load '%s.oso'!\n\n", shaderName.c_str());
    }

    std::vector<OSL::ustring> outputs;
    outputs.push_back(OSL::ustring("my_layer.Cout"));
    shadingSystem->attribute(state->shaderGroup.get(), "renderer_outputs",
                             OSL::TypeDesc(OSL::TypeDesc::STRING, (int)outputs.size()), 
                             &outputs[0]);

    bool groupSuccess = shadingSystem->ShaderGroupEnd(*(state->shaderGroup));
    if (!groupSuccess) {
        fprintf(stderr, "\n[OSL ERROR] ShaderGroup linking failed! Check terminal.\n\n");
    }

    // Force JIT compilation before lookup
    // Without this, the final symbol table doesn't exist yet, and find_symbol will fail
    shadingSystem->optimize_all_groups();

    // Symbol lookup
    state->coutSymbol = shadingSystem->find_symbol(*(state->shaderGroup), OSL::ustring("my_layer.Cout"));
    if (!state->coutSymbol) {
        state->coutSymbol = shadingSystem->find_symbol(*(state->shaderGroup), OSL::ustring("Cout"));
    }

    if (!state->coutSymbol) {
        fprintf(stderr, "[OSL ERROR] Could not find output symbol in %s.\n", shaderName.c_str());
    } else {
        printf("[OSL SUCCESS] Successfully linked shader: %s\n", shaderName.c_str());
    }
}

SampledSpectrum OSLSpectrumTexture::Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
    if (!state || !state->shaderGroup || !shadingSystem || !state->coutSymbol) return SampledSpectrum(0.f);

    OSL::ShaderGlobals sg;
    memset(&sg, 0, sizeof(OSL::ShaderGlobals)); 
    
    sg.P = OSL::Vec3(ctx.p.x, ctx.p.y, ctx.p.z); 
    sg.N = OSL::Vec3(ctx.n.x, ctx.n.y, ctx.n.z);
    sg.Ng = sg.N;
    sg.dPdx = OSL::Vec3(ctx.dpdx.x, ctx.dpdx.y, ctx.dpdx.z);
    sg.dPdy = OSL::Vec3(ctx.dpdy.x, ctx.dpdy.y, ctx.dpdy.z);
    sg.u = ctx.uv[0];
    sg.v = ctx.uv[1];

    sg.dudx = ctx.dudx;
    sg.dudy = ctx.dudy;
    sg.dvdx = ctx.dvdx;
    sg.dvdy = ctx.dvdy;

    thread_local OSL::PerThreadInfo* threadInfo = nullptr;
    if (!threadInfo) threadInfo = shadingSystem->create_thread_info();
    OSL::ShadingContext* shadingCtx = shadingSystem->get_context(threadInfo);
    
    bool success = shadingSystem->execute(*shadingCtx, *(state->shaderGroup), sg);
    
    RGB rgb(0, 0, 0);
    if (success) {
        const float* colorData = (const float*)shadingSystem->symbol_address(*shadingCtx, state->coutSymbol); 
        if (colorData) {
            rgb = RGB(colorData[0], colorData[1], colorData[2]);
        }
    }
    shadingSystem->release_context(shadingCtx);

    // Convert OSL RGB into pbrt Spectrum
    const RGBColorSpace *sRGB = RGBColorSpace::sRGB;
    if (spectrumType == SpectrumType::Unbounded)
        return RGBUnboundedSpectrum(*sRGB, rgb).Sample(lambda);
    else if (spectrumType == SpectrumType::Albedo)
        return RGBAlbedoSpectrum(*sRGB, Clamp(rgb, 0, 1)).Sample(lambda);
    else
        return RGBIlluminantSpectrum(*sRGB, rgb).Sample(lambda);
}

std::string OSLSpectrumTexture::ToString() const {
    return "[ OSLSpectrumTexture ]";
}

} // namespace pbrt