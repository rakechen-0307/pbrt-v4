#include <vector>
#include <pbrt/textures.h>
#include <cstdlib>
#include <string>

// Hardware intrinsics must load before OSL
#include <immintrin.h>
#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>
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

void InitOSL() {
    if (!oslServices) {
        oslServices = new PbrtRendererServices();
        static OIIO::ErrorHandler errHandler; 
        shadingSystem = new OSL::ShadingSystem(oslServices, nullptr, &errHandler);
        
        // Read the OSL_SHADER_PATH environment variable to set the shader search path
        // Start with the current directory as a fallback
        std::string shaderSearchPath = ".";

        if (const char* envPath = std::getenv("OSL_SHADER_PATH")) {
        #ifdef _WIN32
            shaderSearchPath += std::string(";") + envPath;
        #else
            shaderSearchPath += std::string(":") + envPath;
        #endif
        }
        
        shadingSystem->attribute("searchpath:shader", shaderSearchPath);
        printf("[OSL] Shader search path configured as: %s\n", shaderSearchPath.c_str());
    }
}

void CleanupOSL() {
    if (shadingSystem) {
        delete shadingSystem;
        shadingSystem = nullptr;
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

OSLFloatTexture::OSLFloatTexture(const std::string& shaderName) {
    state = new OSLTextureState();
    if (!shadingSystem) return;

    // Turn off OSL's optimizer entirely
    // This guarantees OSL will not delete our variables behind our backs.
    int opt = 0;
    shadingSystem->attribute("optimize", opt);

    state->shaderGroup = shadingSystem->ShaderGroupBegin(shaderName);
    
    bool shaderSuccess = shadingSystem->Shader(*(state->shaderGroup), "surface", shaderName, "my_layer");
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
    // Without this, the final symbol table doesn't exist yet, and find_symbol will fail.
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

} // namespace pbrt