#import <Cocoa/Cocoa.h>
#import <VideoToolbox/VideoToolbox.h>
#import <CoreMedia/CoreMedia.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/QuartzCore.h>
#import "smooth.h"

static const NSInteger kProcessWidth = 1280, kProcessHeight = 720;
static const NSInteger kFullWidth = 1920, kFullHeight = 1080;
// Warm-up calls run back to back, which an M2 Pro does in 14 ms against 20 ms at one call per frame.
static const double kFullCallLimit = 0.025;
static const OSType kProcessFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
static const double kCommitLead = 0.013;
static const double kRepeatRatio = 0.3;
static const NSInteger kFlowWidth = 640, kFlowHeight = 360;

static double HostNow(void) { return CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock())); }
static CMTime HostTime(double seconds) { return CMTimeMakeWithSeconds(seconds, 1000000000); }

static CVPixelBufferPoolRef NewPool(NSDictionary *base, NSInteger width, NSInteger height) {
    NSMutableDictionary *attributes = [base mutableCopy] ?: [NSMutableDictionary dictionary];
    attributes[(id)kCVPixelBufferWidthKey] = @(width);
    attributes[(id)kCVPixelBufferHeightKey] = @(height);
    attributes[(id)kCVPixelBufferPixelFormatTypeKey] = @(kProcessFormat);
    attributes[(id)kCVPixelBufferIOSurfacePropertiesKey] = @{};
    attributes[(id)kCVPixelBufferMetalCompatibilityKey] = @YES;
    CVPixelBufferPoolRef pool = NULL;
    CVPixelBufferPoolCreate(NULL, NULL, (__bridge CFDictionaryRef)attributes, &pool);
    return pool;
}

@interface Smoother : NSObject
- (void)submit:(CVPixelBufferRef)frame;
- (void)hide;
@end

@implementation Smoother {
    dispatch_queue_t _queue;
    BOOL _debug, _debugDiff, _prepared, _wantFull, _wantFlow;
    int _failed;

    VTFrameProcessor *_processor;
    CVPixelBufferPoolRef _middlePool;
    CVPixelBufferPoolRef _fullPool;
    VTFrameProcessor *_scaler;
    CVPixelBufferPoolRef _scaledPool;
    VTPixelTransferSessionRef _transfer;

    VTFrameProcessor *_flow;
    OSType _flowInputFormat;
    CVPixelBufferPoolRef _flowInputPool, _flowFieldPool, _warpPool;
    CVPixelBufferRef _previousFlowInput, _currentFlowInput;
    id<MTLRenderPipelineState> _shrinkPipeline, _choosePipeline, _settlePipeline, _warpLumaPipeline, _warpChromaPipeline;
    id<MTLRenderPipelineState> _finishLumaPipeline, _finishChromaPipeline;
    id<MTLTexture> _quarterFirst, _quarterSecond, _chosen, _settled, _crispLuma, _crispChroma, _quarterCrisp;
    CVPixelBufferPoolRef _smallPool;

    id<MTLDevice> _device;
    id<MTLCommandQueue> _commands;
    CVMetalTextureCacheRef _textures, _presentTextures;
    id<MTLRenderPipelineState> _presentPipeline, _sharpenPipeline;
    CAMetalLayer *_layer;
    NSObject *_presentLock;
    id<MTLFXSpatialScaler> _upscaler;
    id<MTLTexture> _converted, _upscaled;
    BOOL _upscale;
    float _sharpen;
    size_t _viewPixelWidth, _viewPixelHeight;
    double _presentSum;
    int _presentCount;

    NSWindow *_host;
    NSPanel *_child;
    BOOL _attached, _hidden;

    NSMutableArray *_pending;
    int _shown, _missed, _dropped, _shifts, _lastOffset, _lateRun, _resynced;
    BOOL _resync;
    int _glass[4];
    double _slotVsync, _period;
    BOOL _slotFilled;

    CVPixelBufferRef _previousFrame, _previousNative;
    double _previousTime;
    uint8_t *_previousSamples;
    size_t _sampleCount;
    int _repeatsInRow;
    double _lastNewMean, _lastRepeatMean, _lastNewTime, _motionLevel;
    BOOL _sceneCut;
    CVPixelBufferRef _restFrame, _restNative;
    double _restTime;
    NSUInteger _generation;

    double _sourceInterval, _lastTarget;

    NSString *_dumpDirectory;
    dispatch_queue_t _dumpQueue;
    int _dumpCount, _dumpCandidates;
    double _lastMean;

    double _statsStart, _interpSum, _scaleSum, _delaySum;
    int _received, _unique, _interpolated, _scaled, _cuts, _holds, _skipped, _delayCount;
}

- (instancetype)init {
    self = [super init];
    _queue = dispatch_queue_create("smooth.frames", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
    _presentLock = [NSObject new];
    _pending = [NSMutableArray array];
    _period = 1.0 / 60.0;
    _sourceInterval = 1.0 / 30.0;
    _debug = getenv("SMOOTH_DEBUG") != NULL;
    _debugDiff = getenv("SMOOTH_DIFF") != NULL;
    _wantFull = getenv("SMOOTH_INTERP_720") == NULL;
    _wantFlow = getenv("SMOOTH_FLOW") != NULL;
    _upscale = !getenv("SMOOTH_UPSCALE") || atoi(getenv("SMOOTH_UPSCALE")) != 0;
    _sharpen = getenv("SMOOTH_SHARPEN") ? fminf(fmaxf(atof(getenv("SMOOTH_SHARPEN")), 0), 1) : 0.7f;
    if (getenv("SMOOTH_DUMP")) {
        _dumpDirectory = @(getenv("SMOOTH_DUMP"));
        _dumpQueue = dispatch_queue_create("smooth.dump", DISPATCH_QUEUE_SERIAL);
    }
    _device = MTLCreateSystemDefaultDevice();
    _commands = [_device newCommandQueue];
    CVMetalTextureCacheCreate(NULL, NULL, _device, NULL, &_textures);
    CVMetalTextureCacheCreate(NULL, NULL, _device, NULL, &_presentTextures);
    return self;
}

- (BOOL)startInterpolatorWidth:(NSInteger)width height:(NSInteger)height {
    VTLowLatencyFrameInterpolationConfiguration *config =
        [[VTLowLatencyFrameInterpolationConfiguration alloc] initWithFrameWidth:width frameHeight:height numberOfInterpolatedFrames:1];
    VTFrameProcessor *processor = [VTFrameProcessor new];
    NSError *error = nil;
    if (!config || ![processor startSessionWithConfiguration:config error:&error]) {
        fprintf(stderr, "smooth: interpolation at %ldx%ld unavailable: %s\n", (long)width, (long)height,
                error ? error.description.UTF8String : "size rejected");
        return NO;
    }
    _processor = processor;
    _middlePool = NewPool(config.destinationPixelBufferAttributes, width, height);
    // Built from the interpolator's own source attributes, which above 1280x720 can ask for padded buffers.
    if (width != kProcessWidth) _fullPool = NewPool(config.sourcePixelBufferAttributes, width, height);
    return YES;
}

- (void)stopInterpolator {
    [_processor endSession];
    _processor = nil;
    CVPixelBufferPoolRelease(_middlePool);
    _middlePool = NULL;
    CVPixelBufferPoolRelease(_fullPool);
    _fullPool = NULL;
}

static CVPixelBufferPoolRef NewPlainPool(OSType format, size_t width, size_t height) {
    NSDictionary *attributes = @{(id)kCVPixelBufferWidthKey: @(width), (id)kCVPixelBufferHeightKey: @(height), (id)kCVPixelBufferPixelFormatTypeKey: @(format),
                                 (id)kCVPixelBufferIOSurfacePropertiesKey: @{}, (id)kCVPixelBufferMetalCompatibilityKey: @YES};
    CVPixelBufferPoolRef pool = NULL;
    CVPixelBufferPoolCreate(NULL, NULL, (__bridge CFDictionaryRef)attributes, &pool);
    return pool;
}

// Optical flow on small copies, then the stream's own pixels moved half way along it: the middle keeps their texture,
// which the 1280x720 interpolation network loses.
- (void)startFlow {
    if (![VTOpticalFlowConfiguration isSupported]) return;
    VTOpticalFlowConfiguration *config = [[VTOpticalFlowConfiguration alloc] initWithFrameWidth:kFlowWidth frameHeight:kFlowHeight
        qualityPrioritization:VTOpticalFlowConfigurationQualityPrioritizationNormal revision:[VTOpticalFlowConfiguration defaultRevision]];
    VTFrameProcessor *flow = [VTFrameProcessor new];
    NSError *error = nil;
    if (!config || ![flow startSessionWithConfiguration:config error:&error]) {
        fprintf(stderr, "smooth: optical flow unavailable: %s\n", error ? error.description.UTF8String : "size rejected");
        return;
    }
    static NSString *const shader = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "struct Out { float4 position [[position]]; float2 uv; };\n"
        "vertex Out fullscreen(uint id [[vertex_id]]) {\n"
        "    float2 p = float2((id << 1) & 2, id & 2);\n"
        "    Out o; o.position = float4(p * 2.0 - 1.0, 0, 1); o.uv = float2(p.x, 1.0 - p.y); return o;\n"
        "}\n"
        "constexpr sampler smooth(address::clamp_to_edge, filter::linear);\n"
        "constexpr sampler exact(address::clamp_to_edge, filter::nearest);\n"
        // A quarter-size copy: four bilinear taps average each 4x4 block.
        "fragment float4 shrink(Out in [[stage_in]], texture2d<float> image [[texture(0)]]) {\n"
        "    float2 texel = 1.0 / float2(image.get_width(), image.get_height());\n"
        "    return 0.25 * (image.sample(smooth, in.uv + texel * float2(-1, -1)) + image.sample(smooth, in.uv + texel * float2(1, -1))\n"
        "                 + image.sample(smooth, in.uv + texel * float2(-1, 1)) + image.sample(smooth, in.uv + texel * float2(1, 1)));\n"
        "}\n"
        // What a motion through this point costs: how unlike the two real frames are at its two ends over a 3x3 grid of
        // taps, plus up to 8 luma levels a tap for departing (by up to 8 pixels) from the flow measured here, less up to
        // 20 levels a tap for structure at both ends. Sky matches sky and a plain panel matches itself under almost any
        // motion; without the last two terms such matches punched holes in thin trunks and erased overlay icons.
        "static float price(texture2d<float> first, texture2d<float> second, float2 uv, float2 motion, float2 here, float2 there, float2 toPixels) {\n"
        "    float2 texel = 2.0 / float2(first.get_width(), first.get_height());\n"
        "    float before[9], after[9];\n"
        "    float unlike = 0.0, beforeMean = 0.0, afterMean = 0.0;\n"
        "    for (int i = 0; i < 9; i++) {\n"
        "        float2 at = uv + float2(i % 3 - 1, i / 3 - 1) * texel;\n"
        "        before[i] = first.sample(smooth, at - 0.5 * motion).r;\n"
        "        after[i] = second.sample(smooth, at + 0.5 * motion).r;\n"
        "        unlike += abs(before[i] - after[i]);\n"
        "        beforeMean += before[i] / 9.0;\n"
        "        afterMean += after[i] / 9.0;\n"
        "    }\n"
        "    float beforeStructure = 0.0, afterStructure = 0.0;\n"
        "    for (int i = 0; i < 9; i++) { beforeStructure += abs(before[i] - beforeMean) / 9.0; afterStructure += abs(after[i] - afterMean) / 9.0; }\n"
        "    float departure = min(distance(motion * toPixels, here * toPixels), distance(motion * toPixels, there * toPixels));\n"
        "    return unlike + 72.0 / 255.0 * min(departure, 8.0) / 8.0 - 4.5 * min(min(beforeStructure, afterStructure), 40.0 / 255.0);\n"
        "}\n"
        // The flow field is coarse and blends motions where a limb crosses the background, which doubled outlines. Each
        // point instead takes, from the motions of the nearby flow cells and no motion at all (overlays), the best match.
        "fragment float4 choose(Out in [[stage_in]], texture2d<float> first [[texture(0)]], texture2d<float> second [[texture(1)]],\n"
        "                       texture2d<float> forward [[texture(2)]], texture2d<float> backward [[texture(3)]],\n"
        "                       constant float &pixelsPerCell [[buffer(0)]]) {\n"
        "    float2 cell = 1.0 / float2(forward.get_width(), forward.get_height());\n"
        "    float2 here = forward.sample(smooth, in.uv).rg * cell, there = -backward.sample(smooth, in.uv).rg * cell;\n"
        "    float2 toPixels = pixelsPerCell / cell;\n"
        "    float2 best = here;\n"
        "    float bestError = price(first, second, in.uv, best, here, there, toPixels);\n"
        "    float error = price(first, second, in.uv, there, here, there, toPixels);\n"
        "    if (error < bestError) { bestError = error; best = there; }\n"
        "    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) for (int side = 0; side < 2; side++) {\n"
        "        float2 at = in.uv + float2(x, y) * cell;\n"
        "        float2 motion = side == 0 ? forward.sample(exact, at).rg * cell : -backward.sample(exact, at).rg * cell;\n"
        "        error = price(first, second, in.uv, motion, here, there, toPixels);\n"
        "        if (error < bestError) { bestError = error; best = motion; }\n"
        "    }\n"
        // A near, thin object moves a few pixels differently from every flow cell around it (a trunk 80 pixels against
        // 88 for what is behind it), and off by that much it does not match itself. Nudging the best motion by 6 and
        // then 3 pixels finds it before "no motion" is tried.
        "    for (int round = 0; round < 2; round++) {\n"
        "        float2 from = best;\n"
        "        float reach = round == 0 ? 6.0 : 3.0;\n"
        "        for (int i = 0; i < 4; i++) {\n"
        "            float2 motion = from + (i < 2 ? float2(i == 0 ? reach : -reach, 0.0) : float2(0.0, i == 2 ? reach : -reach)) / toPixels;\n"
        "            error = price(first, second, in.uv, motion, here, there, toPixels);\n"
        "            if (error < bestError) { bestError = error; best = motion; }\n"
        "        }\n"
        "    }\n"
        "    error = price(first, second, in.uv, float2(0.0), here, there, toPixels);\n"
        "    if (error < bestError) { bestError = error; best = float2(0.0); }\n"
        "    return float4(best, bestError, 1.0);\n"
        "}\n"
        // Removes stray choices by taking the most central of the 3x3 motions around a point. An average would invent
        // motions between a limb's and the background's, and pixels moved by those drew thin lines around the character.
        "fragment float4 settle(Out in [[stage_in]], texture2d<float> motion [[texture(0)]]) {\n"
        "    float2 texel = 1.0 / float2(motion.get_width(), motion.get_height());\n"
        "    float2 around[9];\n"
        "    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) around[(y + 1) * 3 + x + 1] = motion.sample(exact, in.uv + float2(x, y) * texel).rg;\n"
        "    float2 best = around[4];\n"
        "    float bestCost = 1e9;\n"
        "    for (int i = 0; i < 9; i++) {\n"
        "        float cost = 0.0;\n"
        "        for (int j = 0; j < 9; j++) cost += distance(around[i], around[j]);\n"
        "        if (cost < bestCost) { bestCost = cost; best = around[i]; }\n"
        "    }\n"
        "    float cost = 0.0;\n"
        "    for (int y = -2; y <= 2; y++) for (int x = -2; x <= 2; x++) cost += motion.sample(exact, in.uv + float2(x, y) * texel).b / 25.0;\n"
        "    return float4(best, cost, 1.0);\n"
        "}\n"
        // Beside a fast-moving outline only one real frame shows what is there, no motion matches, and the pixels brought
        // in came out as hard fragments. Where the chosen motions cost more than 6 luma levels a tap the picture fades
        // into a blurred copy of itself (fully at 16), which reads as motion blur.
        "fragment float4 finish(Out in [[stage_in]], texture2d<float> crisp [[texture(0)]], texture2d<float> soft [[texture(1)]],\n"
        "                       texture2d<float> chosen [[texture(2)]], constant float &spacing [[buffer(0)]]) {\n"
        "    float4 sharp = crisp.sample(smooth, in.uv);\n"
        "    float doubt = saturate((chosen.sample(smooth, in.uv).b * 255.0 / 9.0 - 6.0) / 10.0);\n"
        "    if (doubt <= 0.0) return sharp;\n"
        "    float2 texel = spacing / float2(soft.get_width(), soft.get_height());\n"
        "    float4 blurred = float4(0.0);\n"
        "    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) blurred += soft.sample(smooth, in.uv + float2(x, y) * texel) / 9.0;\n"
        "    return mix(sharp, blurred, doubt);\n"
        "}\n"
        "fragment float4 warp(Out in [[stage_in]], texture2d<float> first [[texture(0)]], texture2d<float> second [[texture(1)]],\n"
        "                     texture2d<float> chosen [[texture(2)]], texture2d<float> forward [[texture(3)]],\n"
        "                     texture2d<float> backward [[texture(4)]], texture2d<float> firstLuma [[texture(5)]],\n"
        "                     texture2d<float> secondLuma [[texture(6)]], constant float &pixelsPerCell [[buffer(0)]]) {\n"
        "    float2 cell = 1.0 / float2(forward.get_width(), forward.get_height());\n"
        // Blends the motion between the four chosen points around this pixel only where they agree within 2 pixels. Where
        // they do not, the nearest point's motion left 4-pixel steps along outlines, so the pixel takes whichever of the
        // four matches the two real frames best around it, at full size.
        "    float2 size = float2(chosen.get_width(), chosen.get_height());\n"
        "    float2 place = in.uv * size - 0.5, corner = floor(place), part = place - corner;\n"
        "    float2 limit = size - 1.0;\n"
        "    float2 m00 = chosen.read(uint2(clamp(corner, float2(0.0), limit))).rg, m10 = chosen.read(uint2(clamp(corner + float2(1, 0), float2(0.0), limit))).rg;\n"
        "    float2 m01 = chosen.read(uint2(clamp(corner + float2(0, 1), float2(0.0), limit))).rg, m11 = chosen.read(uint2(clamp(corner + float2(1, 1), float2(0.0), limit))).rg;\n"
        "    float2 motion = part.y < 0.5 ? (part.x < 0.5 ? m00 : m10) : (part.x < 0.5 ? m01 : m11);\n"
        "    float2 toPixels = pixelsPerCell / cell;\n"
        "    float spread = max(max(distance(m00 * toPixels, motion * toPixels), distance(m10 * toPixels, motion * toPixels)),\n"
        "                       max(distance(m01 * toPixels, motion * toPixels), distance(m11 * toPixels, motion * toPixels)));\n"
        "    if (spread < 2.0) {\n"
        "        motion = mix(mix(m00, m10, part.x), mix(m01, m11, part.x), part.y);\n"
        "    } else {\n"
        "        float2 nearest = motion, options[4] = {m00, m10, m01, m11};\n"
        "        float2 texel = 1.0 / float2(firstLuma.get_width(), firstLuma.get_height());\n"
        "        float bestError = 1e9;\n"
        "        for (int i = 0; i < 4; i++) {\n"
        "            float error = all(options[i] == nearest) ? 0.0 : 9.0 / 255.0;\n"
        "            for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {\n"
        "                float2 at = in.uv + float2(x, y) * texel;\n"
        "                error += abs(firstLuma.sample(smooth, at - 0.5 * options[i]).r - secondLuma.sample(smooth, at + 0.5 * options[i]).r);\n"
        "            }\n"
        "            if (error < bestError) { bestError = error; motion = options[i]; }\n"
        "        }\n"
        "    }\n"
        "    float2 fromFirst = in.uv - 0.5 * motion, fromSecond = in.uv + 0.5 * motion;\n"
        // A source whose own flow disagrees with the motion is hidden or new in that frame, so the other one counts more.
        "    float2 firstError = (forward.sample(smooth, fromFirst).rg * cell - motion) / cell * pixelsPerCell;\n"
        "    float2 secondError = (-backward.sample(smooth, fromSecond).rg * cell - motion) / cell * pixelsPerCell;\n"
        "    float firstWeight = 1.0 / (1.0 + 0.1 * dot(firstError, firstError)), secondWeight = 1.0 / (1.0 + 0.1 * dot(secondError, secondError));\n"
        "    if (any(fromFirst < 0.0) || any(fromFirst > 1.0)) firstWeight *= 0.001;\n"
        "    if (any(fromSecond < 0.0) || any(fromSecond > 1.0)) secondWeight *= 0.001;\n"
        "    return (firstWeight * first.sample(smooth, fromFirst) + secondWeight * second.sample(smooth, fromSecond)) / (firstWeight + secondWeight);\n"
        "}\n";
    id<MTLLibrary> library = [_device newLibraryWithSource:shader options:nil error:&error];
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"fullscreen"];
    NSArray *stages = @[@[@"shrink", @(MTLPixelFormatR8Unorm)], @[@"choose", @(MTLPixelFormatRGBA16Float)], @[@"settle", @(MTLPixelFormatRGBA16Float)],
                        @[@"warp", @(MTLPixelFormatR8Unorm)], @[@"warp", @(MTLPixelFormatRG8Unorm)],
                        @[@"finish", @(MTLPixelFormatR8Unorm)], @[@"finish", @(MTLPixelFormatRG8Unorm)]];
    NSMutableArray<id<MTLRenderPipelineState>> *pipelines = [NSMutableArray array];
    for (NSArray *stage in stages) {
        descriptor.fragmentFunction = [library newFunctionWithName:stage[0]];
        descriptor.colorAttachments[0].pixelFormat = [stage[1] unsignedIntegerValue];
        id<MTLRenderPipelineState> pipeline = library ? [_device newRenderPipelineStateWithDescriptor:descriptor error:&error] : nil;
        if (pipeline) [pipelines addObject:pipeline];
    }
    if (pipelines.count != stages.count) {
        fprintf(stderr, "smooth: cannot set up the flow warp: %s\n", error.localizedDescription.UTF8String);
        [flow endSession];
        return;
    }
    _shrinkPipeline = pipelines[0];
    _choosePipeline = pipelines[1];
    _settlePipeline = pipelines[2];
    _warpLumaPipeline = pipelines[3];
    _warpChromaPipeline = pipelines[4];
    _finishLumaPipeline = pipelines[5];
    _finishChromaPipeline = pipelines[6];
    NSDictionary *field = config.destinationPixelBufferAttributes;
    _flowInputFormat = config.frameSupportedPixelFormats.firstObject.unsignedIntValue;
    _flowInputPool = NewPlainPool(_flowInputFormat, kFlowWidth, kFlowHeight);
    _flowFieldPool = NewPlainPool(kCVPixelFormatType_TwoComponent16Half, [field[(id)kCVPixelBufferWidthKey] unsignedLongValue],
                                  [field[(id)kCVPixelBufferHeightKey] unsignedLongValue]);
    _flow = flow;
}

- (CVPixelBufferRef)flowInput:(CVPixelBufferRef)frame {
    CVPixelBufferRef input = NULL;
    CVPixelBufferPoolCreatePixelBuffer(NULL, _flowInputPool, &input);
    if (input && VTPixelTransferSessionTransferImage(_transfer, frame, input) != noErr) { CVPixelBufferRelease(input); input = NULL; }
    return input;
}

- (id<MTLTexture>)texture:(CVPixelBufferRef)buffer plane:(size_t)plane format:(MTLPixelFormat)format keep:(NSMutableArray *)keep {
    CVMetalTextureRef texture = NULL;
    BOOL planar = CVPixelBufferIsPlanar(buffer);
    size_t width = planar ? CVPixelBufferGetWidthOfPlane(buffer, plane) : CVPixelBufferGetWidth(buffer);
    size_t height = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer);
    CVMetalTextureCacheCreateTextureFromImage(NULL, _textures, buffer, NULL, format, width, height, plane, &texture);
    if (!texture) return nil;
    [keep addObject:(__bridge_transfer id)texture];
    return CVMetalTextureGetTexture((__bridge CVMetalTextureRef)keep.lastObject);
}

- (CVPixelBufferRef)flowMiddleFrom:(CVPixelBufferRef)previous at:(double)previousTime to:(CVPixelBufferRef)current at:(double)currentTime {
    OSType format = CVPixelBufferGetPixelFormatType(current);
    size_t width = CVPixelBufferGetWidth(current), height = CVPixelBufferGetHeight(current);
    if (CVPixelBufferGetPixelFormatType(previous) != format || CVPixelBufferGetWidth(previous) != width || CVPixelBufferGetHeight(previous) != height) return NULL;
    if (!_previousFlowInput) _previousFlowInput = [self flowInput:previous];
    CVPixelBufferRelease(_currentFlowInput);
    _currentFlowInput = [self flowInput:current];
    CVPixelBufferRef forward = NULL, backward = NULL, destination = NULL;
    CVPixelBufferPoolCreatePixelBuffer(NULL, _flowFieldPool, &forward);
    CVPixelBufferPoolCreatePixelBuffer(NULL, _flowFieldPool, &backward);
    if (_warpPool) {
        NSDictionary *made = (__bridge NSDictionary *)CVPixelBufferPoolGetPixelBufferAttributes(_warpPool);
        if ([made[(id)kCVPixelBufferPixelFormatTypeKey] unsignedIntValue] != format || [made[(id)kCVPixelBufferWidthKey] unsignedLongValue] != width
            || [made[(id)kCVPixelBufferHeightKey] unsignedLongValue] != height) { CVPixelBufferPoolRelease(_warpPool); _warpPool = NULL; }
    }
    if (!_warpPool) _warpPool = NewPlainPool(format, width, height);
    CVPixelBufferPoolCreatePixelBuffer(NULL, _warpPool, &destination);

    BOOL done = NO;
    if (_previousFlowInput && _currentFlowInput && forward && backward && destination) {
        VTFrameProcessorFrame *source = [[VTFrameProcessorFrame alloc] initWithBuffer:_previousFlowInput presentationTimeStamp:HostTime(previousTime)];
        VTFrameProcessorFrame *next = [[VTFrameProcessorFrame alloc] initWithBuffer:_currentFlowInput presentationTimeStamp:HostTime(currentTime)];
        VTFrameProcessorOpticalFlow *field = [[VTFrameProcessorOpticalFlow alloc] initWithForwardFlow:forward backwardFlow:backward];
        VTOpticalFlowParameters *parameters = (source && next && field) ?
            [[VTOpticalFlowParameters alloc] initWithSourceFrame:source nextFrame:next submissionMode:VTOpticalFlowParametersSubmissionModeRandom
                                          destinationOpticalFlow:field] : nil;
        NSError *error = nil;
        if (parameters && [_flow processWithParameters:parameters error:&error]) {
            NSMutableArray *keep = [NSMutableArray array];
            id<MTLTexture> forwardTexture = [self texture:forward plane:0 format:MTLPixelFormatRG16Float keep:keep];
            id<MTLTexture> backwardTexture = [self texture:backward plane:0 format:MTLPixelFormatRG16Float keep:keep];
            id<MTLCommandBuffer> commands = [_commands commandBuffer];
            if (!_chosen || _chosen.width != width / 4 || _chosen.height != height / 4) {
                MTLTextureDescriptor *small = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:width / 4 height:height / 4 mipmapped:NO];
                small.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
                small.storageMode = MTLStorageModePrivate;
                _quarterFirst = [_device newTextureWithDescriptor:small];
                _quarterSecond = [_device newTextureWithDescriptor:small];
                _quarterCrisp = [_device newTextureWithDescriptor:small];
                small.pixelFormat = MTLPixelFormatRGBA16Float;
                _chosen = [_device newTextureWithDescriptor:small];
                _settled = [_device newTextureWithDescriptor:small];
                small.pixelFormat = MTLPixelFormatR8Unorm;
                small.width = width;
                small.height = height;
                _crispLuma = [_device newTextureWithDescriptor:small];
                small.pixelFormat = MTLPixelFormatRG8Unorm;
                small.width = width / 2;
                small.height = height / 2;
                _crispChroma = [_device newTextureWithDescriptor:small];
            }
            id<MTLTexture> firstLuma = nil, secondLuma = nil;
            BOOL drawn = forwardTexture && backwardTexture && _quarterFirst && _quarterSecond && _quarterCrisp && _chosen && _settled && _crispLuma && _crispChroma;
            for (size_t plane = 0; plane < 2 && drawn; plane++) {
                MTLPixelFormat planeFormat = plane ? MTLPixelFormatRG8Unorm : MTLPixelFormatR8Unorm;
                id<MTLTexture> first = [self texture:previous plane:plane format:planeFormat keep:keep];
                id<MTLTexture> second = [self texture:current plane:plane format:planeFormat keep:keep];
                id<MTLTexture> target = [self texture:destination plane:plane format:planeFormat keep:keep];
                drawn = first && second && target;
                if (plane == 0) { firstLuma = first; secondLuma = second; }
                if (drawn && plane == 0) {
                    [self draw:_shrinkPipeline into:_quarterFirst from:@[first] value:0 commands:commands];
                    [self draw:_shrinkPipeline into:_quarterSecond from:@[second] value:0 commands:commands];
                    [self draw:_choosePipeline into:_chosen from:@[_quarterFirst, _quarterSecond, forwardTexture, backwardTexture]
                         value:(float)width / (float)forwardTexture.width commands:commands];
                    [self draw:_settlePipeline into:_settled from:@[_chosen] value:0 commands:commands];
                }
                if (drawn) {
                    id<MTLTexture> crisp = plane ? _crispChroma : _crispLuma;
                    [self draw:(plane ? _warpChromaPipeline : _warpLumaPipeline) into:crisp
                          from:@[first, second, _settled, forwardTexture, backwardTexture, firstLuma, secondLuma]
                         value:(float)width / (float)forwardTexture.width commands:commands];
                    if (plane == 0) [self draw:_shrinkPipeline into:_quarterCrisp from:@[crisp] value:0 commands:commands];
                    [self draw:(plane ? _finishChromaPipeline : _finishLumaPipeline) into:target from:@[crisp, plane ? crisp : _quarterCrisp, _settled]
                         value:(plane ? 3.0f : 1.5f) commands:commands];
                }
            }
            if (drawn) {
                [commands commit];
                [commands waitUntilCompleted];
                CVBufferPropagateAttachments(current, destination);
                done = commands.status == MTLCommandBufferStatusCompleted;
            }
        } else if (_failed++ == 0) {
            fprintf(stderr, "smooth: optical flow failed: %s\n", error ? error.description.UTF8String : "frames rejected");
        }
    }
    CVPixelBufferRelease(forward);
    CVPixelBufferRelease(backward);
    if (!done) { CVPixelBufferRelease(destination); destination = NULL; }
    return destination;
}

// Runs a few calls on blank frames so the first real ones are not slow, and returns the fastest warm call in seconds.
- (double)warmUp {
    CVPixelBufferPoolRef inputPool = _fullPool ?: _smallPool;
    CVPixelBufferRef blankA = NULL, blankB = NULL;
    CVPixelBufferPoolCreatePixelBuffer(NULL, inputPool, &blankA);
    CVPixelBufferPoolCreatePixelBuffer(NULL, inputPool, &blankB);
    double fastest = 0;
    for (int round = 0; round < 3 && blankA && blankB; round++) {
        double begin = HostNow();
        CVPixelBufferRef middle = [self interpolateFrom:blankA at:0 to:blankB at:1.0 / 30];
        double took = HostNow() - begin;
        if (round > 0 && (fastest == 0 || took < fastest)) fastest = took;
        CVPixelBufferRef finished = [self finished:middle ?: blankB large:YES];
        CVPixelBufferRelease(finished);
        CVPixelBufferRelease(middle);
    }
    CVPixelBufferRelease(blankA);
    CVPixelBufferRelease(blankB);
    return fastest;
}

- (void)prepare {
    if (_prepared) return;
    _prepared = YES;
    BOOL supported = [VTLowLatencyFrameInterpolationConfiguration isSupported];
    // Above 1280x720's pixels the interpolator switches to a slower network whose middle frames are closer to the real ones.
    if (supported && _wantFull && !_wantFlow) [self startInterpolatorWidth:kFullWidth height:kFullHeight];
    if (supported && !_processor) [self startInterpolatorWidth:kProcessWidth height:kProcessHeight];
    if ([VTLowLatencySuperResolutionScalerConfiguration isSupported]) {
        NSArray<NSNumber *> *factors = [VTLowLatencySuperResolutionScalerConfiguration supportedScaleFactorsForFrameWidth:kProcessWidth frameHeight:kProcessHeight];
        if ([factors containsObject:@1.5f]) {
            VTLowLatencySuperResolutionScalerConfiguration *config =
                [[VTLowLatencySuperResolutionScalerConfiguration alloc] initWithFrameWidth:kProcessWidth frameHeight:kProcessHeight scaleFactor:1.5f];
            VTFrameProcessor *scaler = [VTFrameProcessor new];
            if ([scaler startSessionWithConfiguration:config error:nil]) {
                _scaler = scaler;
                _scaledPool = NewPool(config.destinationPixelBufferAttributes, kProcessWidth * 3 / 2, kProcessHeight * 3 / 2);
            }
        }
    }
    VTPixelTransferSessionCreate(NULL, &_transfer);
    _smallPool = NewPool(nil, kProcessWidth, kProcessHeight);
    if (_wantFlow) [self startFlow];

    double call = [self warmUp];
    // A slower Mac (an M1 takes about 100 ms at 1080p) would miss nearly every middle frame, so it keeps 1280x720.
    if (_fullPool && call > kFullCallLimit) {
        fprintf(stderr, "smooth: interpolation at 1920x1080 takes %.0f ms on this Mac, using 1280x720\n", call * 1000);
        [self stopInterpolator];
        if (supported) [self startInterpolatorWidth:kProcessWidth height:kProcessHeight];
        call = [self warmUp];
    }
    if (_debug) fprintf(stderr, "smooth: interpolating at %s, %.1f ms a call\n", _fullPool ? "1920x1080" : "1280x720", call * 1000);
    if (_flow) {
        CVPixelBufferRef blankA = NULL, blankB = NULL;
        CVPixelBufferPoolCreatePixelBuffer(NULL, _smallPool, &blankA);
        CVPixelBufferPoolCreatePixelBuffer(NULL, _smallPool, &blankB);
        double fastest = 0;
        for (int round = 0; round < 3 && blankA && blankB; round++) {
            double begin = HostNow();
            CVPixelBufferRelease([self flowMiddleFrom:blankA at:0 to:blankB at:1.0 / 30]);
            if (round > 0 && (fastest == 0 || HostNow() - begin < fastest)) fastest = HostNow() - begin;
            CVPixelBufferRelease(_previousFlowInput);
            _previousFlowInput = NULL;
        }
        CVPixelBufferRelease(_currentFlowInput);
        _currentFlowInput = NULL;
        CVPixelBufferRelease(blankA);
        CVPixelBufferRelease(blankB);
        if (_debug) fprintf(stderr, "smooth: middle frames from optical flow, %.1f ms a call\n", fastest * 1000);
    }
    _failed = 0;
    _scaleSum = 0;
    _scaled = 0;
}

- (CVPixelBufferRef)interpolateFrom:(CVPixelBufferRef)previous at:(double)previousTime to:(CVPixelBufferRef)current at:(double)currentTime {
    if (!_processor) return NULL;
    CVPixelBufferRef destination = NULL;
    CVPixelBufferPoolCreatePixelBuffer(NULL, _middlePool, &destination);
    if (!destination) return NULL;
    CVBufferPropagateAttachments(current, destination);
    VTFrameProcessorFrame *source = [[VTFrameProcessorFrame alloc] initWithBuffer:current presentationTimeStamp:HostTime(currentTime)];
    VTFrameProcessorFrame *reference = [[VTFrameProcessorFrame alloc] initWithBuffer:previous presentationTimeStamp:HostTime(previousTime)];
    VTFrameProcessorFrame *output = [[VTFrameProcessorFrame alloc] initWithBuffer:destination presentationTimeStamp:HostTime((previousTime + currentTime) / 2)];
    VTLowLatencyFrameInterpolationParameters *parameters = (source && reference && output) ?
        [[VTLowLatencyFrameInterpolationParameters alloc] initWithSourceFrame:source previousFrame:reference
                                                           interpolationPhase:@[@0.5f] destinationFrames:@[output]] : nil;
    NSError *error = nil;
    if (!parameters || ![_processor processWithParameters:parameters error:&error]) {
        if (_failed++ == 0) fprintf(stderr, "smooth: interpolation failed: %s\n", error ? error.description.UTF8String : "frames rejected");
        CVPixelBufferRelease(destination);
        return NULL;
    }
    return destination;
}

static void WriteNV12(CVPixelBufferRef buffer, NSString *path) {
    CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    FILE *file = fopen(path.fileSystemRepresentation, "wb");
    if (file) {
        size_t width = CVPixelBufferGetWidth(buffer);
        for (size_t plane = 0; plane < 2; plane++) {
            const uint8_t *base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane);
            size_t stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane);
            size_t rows = CVPixelBufferGetHeightOfPlane(buffer, plane);
            for (size_t row = 0; row < rows; row++) fwrite(base + row * stride, 1, width, file);
        }
        fclose(file);
    }
    CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
}

- (void)dumpPrevious:(CVPixelBufferRef)previous middle:(CVPixelBufferRef)middle current:(CVPixelBufferRef)current
       nativePrevious:(CVPixelBufferRef)nativePrevious nativeCurrent:(CVPixelBufferRef)nativeCurrent {
    int index = _dumpCount++;
    double mean = _lastMean;
    NSString *directory = _dumpDirectory;
    CVPixelBufferRetain(previous);
    CVPixelBufferRetain(middle);
    CVPixelBufferRetain(current);
    CVPixelBufferRetain(nativePrevious);
    CVPixelBufferRetain(nativeCurrent);
    dispatch_async(_dumpQueue, ^{
        NSString *size = [NSString stringWithFormat:@"%zux%zu", CVPixelBufferGetWidth(middle), CVPixelBufferGetHeight(middle)];
        WriteNV12(previous, [directory stringByAppendingFormat:@"/%02d_a_%@.nv12", index, size]);
        WriteNV12(middle, [directory stringByAppendingFormat:@"/%02d_m_%@.nv12", index, size]);
        WriteNV12(current, [directory stringByAppendingFormat:@"/%02d_b_%@.nv12", index, size]);
        if (nativePrevious && nativePrevious != previous) {
            NSString *nativeSize = [NSString stringWithFormat:@"%zux%zu", CVPixelBufferGetWidth(nativeCurrent), CVPixelBufferGetHeight(nativeCurrent)];
            WriteNV12(nativePrevious, [directory stringByAppendingFormat:@"/%02d_na_%@.nv12", index, nativeSize]);
            WriteNV12(nativeCurrent, [directory stringByAppendingFormat:@"/%02d_nb_%@.nv12", index, nativeSize]);
        }
        fprintf(stderr, "smooth: dumped %02d (mean %.2f, %s)\n", index, mean, size.UTF8String);
        CVPixelBufferRelease(nativePrevious);
        CVPixelBufferRelease(nativeCurrent);
        CVPixelBufferRelease(previous);
        CVPixelBufferRelease(middle);
        CVPixelBufferRelease(current);
    });
}

- (CVPixelBufferRef)finished:(CVPixelBufferRef)buffer large:(BOOL)large {
    double begin = HostNow();
    CVPixelBufferRef result = CVPixelBufferRetain(buffer);
    if (large && _scaler && CVPixelBufferGetWidth(buffer) == (size_t)kProcessWidth) {
        CVPixelBufferRef destination = NULL;
        CVPixelBufferPoolCreatePixelBuffer(NULL, _scaledPool, &destination);
        if (destination) {
            CVBufferPropagateAttachments(buffer, destination);
            VTFrameProcessorFrame *source = [[VTFrameProcessorFrame alloc] initWithBuffer:buffer presentationTimeStamp:HostTime(begin)];
            VTFrameProcessorFrame *output = [[VTFrameProcessorFrame alloc] initWithBuffer:destination presentationTimeStamp:HostTime(begin)];
            VTLowLatencySuperResolutionScalerParameters *parameters = (source && output) ?
                [[VTLowLatencySuperResolutionScalerParameters alloc] initWithSourceFrame:source destinationFrame:output] : nil;
            if (parameters && [_scaler processWithParameters:parameters error:nil]) {
                CVPixelBufferRelease(result);
                result = destination;
            } else {
                CVPixelBufferRelease(destination);
            }
        }
    }
    if (result != buffer) { _scaleSum += HostNow() - begin; _scaled++; }
    return result;
}

- (void)attach {
    NSWindow *host = nil;
    for (NSWindow *window in NSApp.windows) {
        if (!window.isVisible || [window isKindOfClass:NSPanel.class] || !window.contentView) continue;
        if (!host || window.frame.size.width * window.frame.size.height > host.frame.size.width * host.frame.size.height) host = window;
    }
    if (!host) {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.5 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{ [self attach]; });
        return;
    }
    NSRect content = [host convertRectToScreen:host.contentLayoutRect];
    NSPanel *child = [[NSPanel alloc] initWithContentRect:content styleMask:NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
                                                  backing:NSBackingStoreBuffered defer:NO];
    child.backgroundColor = NSColor.blackColor;
    child.ignoresMouseEvents = YES;
    child.hasShadow = NO;
    child.hidesOnDeactivate = NO;
    child.releasedWhenClosed = NO;
    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, content.size.width, content.size.height)];
    view.wantsLayer = YES;
    view.layer.backgroundColor = CGColorGetConstantColor(kCGColorBlack);
    child.contentView = view;
    [host addChildWindow:child ordered:NSWindowAbove];
    _host = host;
    _child = child;
    if (_hidden) [child orderOut:nil];
    for (NSNotificationName name in @[NSWindowDidResizeNotification, NSWindowDidEnterFullScreenNotification, NSWindowDidExitFullScreenNotification,
                                      NSWindowDidChangeScreenNotification]) {
        [NSNotificationCenter.defaultCenter addObserverForName:name object:host queue:NSOperationQueue.mainQueue usingBlock:^(NSNotification *note) {
            [child setFrame:[host convertRectToScreen:host.contentLayoutRect] display:YES];
        }];
    }
    for (NSNotificationName name in @[NSWindowDidResizeNotification, NSWindowDidChangeBackingPropertiesNotification]) {
        [NSNotificationCenter.defaultCenter addObserverForName:name object:child queue:NSOperationQueue.mainQueue usingBlock:^(NSNotification *note) {
            [self updateOutputSize];
        }];
    }
    [self startOutputIn:view];
    [self updateOutputSize];
}

- (void)updateOutputSize {
    CGFloat scale = _child.backingScaleFactor;
    NSSize size = _child.contentView.bounds.size;
    @synchronized (_presentLock) {
        _viewPixelWidth = (size_t)lround(size.width * scale);
        _viewPixelHeight = (size_t)lround(size.height * scale);
    }
}

- (void)startOutputIn:(NSView *)view {
    static NSString *const shader = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "struct Out { float4 position [[position]]; float2 uv; };\n"
        "vertex Out fullscreen(uint id [[vertex_id]]) {\n"
        "    float2 p = float2((id << 1) & 2, id & 2);\n"
        "    Out o; o.position = float4(p * 2.0 - 1.0, 0, 1); o.uv = float2(p.x, 1.0 - p.y); return o;\n"
        "}\n"
        "fragment float4 ycbcr(Out in [[stage_in]], texture2d<float> luma [[texture(0)]], texture2d<float> chroma [[texture(1)]],\n"
        "                      constant float &fullRange [[buffer(0)]]) {\n"
        "    constexpr sampler s(address::clamp_to_edge, filter::linear);\n"
        "    float y = luma.sample(s, in.uv).r;\n"
        "    float2 c = chroma.sample(s, in.uv).rg - 0.5;\n"
        "    if (fullRange < 0.5) { y = (y - 16.0 / 255.0) * (255.0 / 219.0); c *= 255.0 / 224.0; }\n"
        "    return float4(y + 1.5748 * c.y, y - 0.1873 * c.x - 0.4681 * c.y, y + 1.8556 * c.x, 1.0);\n"
        "}\n"
        // Contrast-adaptive sharpening (after AMD CAS): flat and already-contrasty pixels get less, so edges do not ring.
        "fragment float4 sharpen(Out in [[stage_in]], texture2d<float> image [[texture(0)]], constant float &strength [[buffer(0)]]) {\n"
        "    uint2 p = uint2(in.position.xy), last = uint2(image.get_width() - 1, image.get_height() - 1);\n"
        "    float3 center = image.read(p).rgb;\n"
        "    if (strength <= 0.0) return float4(center, 1.0);\n"
        "    float3 up = image.read(uint2(p.x, p.y > 0 ? p.y - 1 : 0)).rgb, down = image.read(uint2(p.x, min(p.y + 1, last.y))).rgb;\n"
        "    float3 left = image.read(uint2(p.x > 0 ? p.x - 1 : 0, p.y)).rgb, right = image.read(uint2(min(p.x + 1, last.x), p.y)).rgb;\n"
        "    float3 low = min(center, min(min(up, down), min(left, right)));\n"
        "    float3 high = max(center, max(max(up, down), max(left, right)));\n"
        "    float3 amount = sqrt(saturate(min(low, 1.0 - high) / max(high, 1e-4)));\n"
        "    float3 weight = amount * (-1.0 / mix(8.0, 5.0, strength));\n"
        "    return float4(saturate(((up + down + left + right) * weight + center) / (1.0 + 4.0 * weight)), 1.0);\n"
        "}\n";
    NSError *error = nil;
    id<MTLLibrary> library = [_device newLibraryWithSource:shader options:nil error:&error];
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"fullscreen"];
    descriptor.fragmentFunction = [library newFunctionWithName:@"ycbcr"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    _presentPipeline = library ? [_device newRenderPipelineStateWithDescriptor:descriptor error:&error] : nil;
    if (!_presentPipeline) { fprintf(stderr, "smooth: cannot set up the output: %s\n", error.localizedDescription.UTF8String); return; }
    descriptor.fragmentFunction = [library newFunctionWithName:@"sharpen"];
    _sharpenPipeline = [_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (!_sharpenPipeline) _upscale = NO;

    CAMetalLayer *layer = [CAMetalLayer layer];
    layer.device = _device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    layer.displaySyncEnabled = YES;
    layer.maximumDrawableCount = 3;
    layer.allowsNextDrawableTimeout = YES;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceITUR_709);
    layer.colorspace = colorSpace;
    CGColorSpaceRelease(colorSpace);
    layer.frame = view.bounds;
    layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
    layer.contentsGravity = kCAGravityResizeAspect;
    layer.backgroundColor = CGColorGetConstantColor(kCGColorBlack);
    layer.drawableSize = CGSizeMake(kProcessWidth * 3 / 2, kProcessHeight * 3 / 2);
    [view.layer addSublayer:layer];
    @synchronized (_presentLock) { _layer = layer; }

    CADisplayLink *link = [view displayLinkWithTarget:self selector:@selector(refresh:)];
    NSThread *thread = [[NSThread alloc] initWithBlock:^{
        [link addToRunLoop:NSRunLoop.currentRunLoop forMode:NSDefaultRunLoopMode];
        [NSRunLoop.currentRunLoop run];
    }];
    thread.name = @"smooth.display";
    thread.qualityOfService = NSQualityOfServiceUserInteractive;
    [thread start];
}

- (void)hide {
    _hidden = YES;
    [_child orderOut:nil];
}

- (void)unhide {
    if (!_hidden) return;
    _hidden = NO;
    if (_child && !_child.parentWindow) [_host addChildWindow:_child ordered:NSWindowAbove];
    [_child orderFront:nil];
}

- (BOOL)isDuplicate:(CVPixelBufferRef)buffer received:(double)received {
    CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    const uint8_t *base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0);
    size_t stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0);
    size_t width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer), step = 8;
    size_t count = (width / step) * (height / step);
    BOOL comparable = _previousSamples && _sampleCount == count;
    if (!comparable) { free(_previousSamples); _previousSamples = malloc(count); _sampleCount = count; }
    uint64_t total = 0;
    int strong = 0;
    size_t index = 0;
    for (size_t y = 0; y < height / step; y++) {
        const uint8_t *row = base + y * step * stride;
        for (size_t x = 0; x < width / step; x++, index++) {
            uint8_t value = row[x * step];
            int difference = comparable ? abs((int)value - (int)_previousSamples[index]) : 0;
            total += (uint64_t)difference;
            if (difference >= 8) strong++;
            _previousSamples[index] = value;
        }
    }
    CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    double mean = (double)total / (double)count;
    _lastMean = mean;
    double now = HostNow();
    if (now - _lastNewTime > 0.15) _lastNewMean = 0;
    double diagNew = _lastNewMean, diagRepeat = _lastRepeatMean;
    int diagRun = _repeatsInRow;
    BOOL identical = mean < 0.01 && strong == 0;
    BOOL refinement = NO;
    if (_repeatsInRow == 0) refinement = mean < kRepeatRatio * _lastNewMean;
    else if (_repeatsInRow == 1) refinement = mean < MAX(2.5 * _lastRepeatMean, 0.08 * _lastNewMean);
    BOOL duplicate = comparable && (identical || refinement);
    _repeatsInRow = duplicate ? _repeatsInRow + 1 : 0;
    if (duplicate) _lastRepeatMean = mean; else { _lastNewMean = mean; _lastNewTime = now; }
    _sceneCut = comparable && !duplicate && mean > 12.0 && mean > 2.5 * _motionLevel;
    if (!duplicate) _motionLevel += 0.2 * (mean - _motionLevel);
    if (_debugDiff) fprintf(stderr, "diff t %.4f mean %.3f strong %d last_new %.3f last_repeat %.3f run %d %s\n", received, mean, strong,
                            diagNew, diagRepeat, diagRun, duplicate ? "repeat" : _sceneCut ? "cut" : "new");
    return duplicate;
}

- (void)show:(CVPixelBufferRef)buffer target:(double)target {
    NSArray *item = @[(__bridge id)buffer, @(target)];
    @synchronized (_pending) {
        if (target > _slotVsync + _period / 2) {
            NSUInteger index = _pending.count;
            while (index > 0 && [_pending[index - 1][1] doubleValue] > target) index--;
            [_pending insertObject:item atIndex:index];
            while (_pending.count > 6) { [_pending removeObjectAtIndex:0]; _missed++; }
            return;
        }
        BOOL inTime = !_slotFilled && target > _slotVsync - _period / 2 && _slotVsync - HostNow() > kCommitLead - 0.001;
        if (!inTime) {
            if (_pending.count == 0) [_pending addObject:@[(__bridge id)buffer, @(_slotVsync + _period)]];
            else _missed++;
            return;
        }
        _slotFilled = YES;
        _shown++;
    }
    [self present:item];
}

- (void)refresh:(CADisplayLink *)link {
    NSArray *next = nil;
    @synchronized (_pending) {
        double target = link.targetTimestamp;
        if (target - link.timestamp > 0.002) _period = target - link.timestamp;
        while (_pending.count && [_pending.firstObject[1] doubleValue] < target - _period / 2) { [_pending removeObjectAtIndex:0]; _missed++; }
        if (_pending.count && [_pending.firstObject[1] doubleValue] < target + _period / 2) {
            next = _pending.firstObject;
            [_pending removeObjectAtIndex:0];
            _shown++;
        }
        _slotFilled = next != nil;
        _slotVsync = target;
    }
    if (next) [self present:next];
}

// Fits the frame into the view's pixels; returns NO when MetalFX is off or would not enlarge the frame.
- (BOOL)prepareUpscaleWidth:(size_t)width height:(size_t)height output:(CGSize *)output {
    if (!_upscale || !_viewPixelWidth || !_viewPixelHeight) return NO;
    size_t fitWidth = _viewPixelWidth, fitHeight = _viewPixelWidth * height / width;
    if (fitHeight > _viewPixelHeight) { fitHeight = _viewPixelHeight; fitWidth = _viewPixelHeight * width / height; }
    if (fitWidth <= width || fitHeight <= height) return NO;
    // Past 2x MetalFX adds no detail, and at a 6016-pixel-wide backing it took 8–13 ms a frame next to the interpolator.
    if (fitWidth > 2 * width) { fitWidth = 2 * width; fitHeight = 2 * height; }
    *output = CGSizeMake(fitWidth, fitHeight);
    if (_upscaler && _upscaler.inputWidth == width && _upscaler.inputHeight == height
        && _upscaler.outputWidth == fitWidth && _upscaler.outputHeight == fitHeight) return YES;
    MTLFXSpatialScalerDescriptor *descriptor = [MTLFXSpatialScalerDescriptor new];
    descriptor.inputWidth = width;
    descriptor.inputHeight = height;
    descriptor.outputWidth = fitWidth;
    descriptor.outputHeight = fitHeight;
    descriptor.colorTextureFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.outputTextureFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
    _upscaler = [descriptor newSpatialScalerWithDevice:_device];
    if (!_upscaler) {
        fprintf(stderr, "smooth: MetalFX cannot scale %zux%zu to %zux%zu, using plain scaling\n", width, height, fitWidth, fitHeight);
        _upscale = NO;
        return NO;
    }
    MTLTextureDescriptor *texture = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:width height:height mipmapped:NO];
    texture.usage = _upscaler.colorTextureUsage | MTLTextureUsageRenderTarget;
    texture.storageMode = MTLStorageModePrivate;
    _converted = [_device newTextureWithDescriptor:texture];
    texture = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:fitWidth height:fitHeight mipmapped:NO];
    texture.usage = _upscaler.outputTextureUsage | MTLTextureUsageShaderRead;
    texture.storageMode = MTLStorageModePrivate;
    _upscaled = [_device newTextureWithDescriptor:texture];
    if (_debug) fprintf(stderr, "smooth: MetalFX scaling %zux%zu to %zux%zu, sharpen %.2f\n", width, height, fitWidth, fitHeight, _sharpen);
    return YES;
}

- (void)draw:(id<MTLRenderPipelineState>)pipeline into:(id<MTLTexture>)target from:(NSArray<id<MTLTexture>> *)sources
       value:(float)value commands:(id<MTLCommandBuffer>)commands {
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    for (NSUInteger index = 0; index < sources.count; index++) [encoder setFragmentTexture:sources[index] atIndex:index];
    [encoder setFragmentBytes:&value length:sizeof(value) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
}

- (void)present:(NSArray *)item {
    @synchronized (_pending) {
        if (_resync) { _resync = NO; _resynced++; return; }
    }
    @synchronized (_presentLock) {
        if (!_layer) return;
        CVPixelBufferRef buffer = (__bridge CVPixelBufferRef)item[0];
        size_t width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer);
        if (!CVPixelBufferIsPlanar(buffer)) return;
        CGSize output = CGSizeMake(width, height);
        BOOL upscale = [self prepareUpscaleWidth:width height:height output:&output];
        if (!CGSizeEqualToSize(_layer.drawableSize, output)) _layer.drawableSize = output;
        CVMetalTextureRef luma = NULL, chroma = NULL;
        CVMetalTextureCacheCreateTextureFromImage(NULL, _presentTextures, buffer, NULL, MTLPixelFormatR8Unorm, width, height, 0, &luma);
        CVMetalTextureCacheCreateTextureFromImage(NULL, _presentTextures, buffer, NULL, MTLPixelFormatRG8Unorm, width / 2, height / 2, 1, &chroma);
        id<CAMetalDrawable> drawable = (luma && chroma) ? [_layer nextDrawable] : nil;
        if (!drawable) {
            if (luma) CFRelease(luma);
            if (chroma) CFRelease(chroma);
            return;
        }
        double target = [item[1] doubleValue];
        [drawable addPresentedHandler:^(id<MTLDrawable> shown) {
            double presented = shown.presentedTime;
            @synchronized (self->_pending) {
                if (presented == 0) { self->_dropped++; return; }
                // A steady offset is compositor latency; judder is the offset changing between frames.
                int offset = (int)MAX(0, MIN(3, lround((presented - target) / self->_period)));
                self->_glass[offset]++;
                if (offset != self->_lastOffset) self->_shifts++;
                self->_lastOffset = offset;
                // After one late frame the layer keeps an extra drawable queued and every later frame arrives a refresh
                // late until a present is left out; half a second of that asks for one.
                self->_lateRun = offset >= 2 ? self->_lateRun + 1 : 0;
                if (self->_lateRun >= 30) { self->_lateRun = 0; self->_resync = YES; }
            }
        }];
        id<MTLCommandBuffer> commands = [_commands commandBuffer];
        float fullRange = CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ? 1 : 0;
        NSArray *planes = @[CVMetalTextureGetTexture(luma), CVMetalTextureGetTexture(chroma)];
        if (upscale) {
            [self draw:_presentPipeline into:_converted from:planes value:fullRange commands:commands];
            _upscaler.colorTexture = _converted;
            _upscaler.outputTexture = _upscaled;
            _upscaler.inputContentWidth = width;
            _upscaler.inputContentHeight = height;
            [_upscaler encodeToCommandBuffer:commands];
            [self draw:_sharpenPipeline into:drawable.texture from:@[_upscaled] value:_sharpen commands:commands];
        } else {
            [self draw:_presentPipeline into:drawable.texture from:planes value:fullRange commands:commands];
        }
        [commands presentDrawable:drawable];
        [commands addCompletedHandler:^(id<MTLCommandBuffer> done) {
            (void)item;
            CFRelease(luma);
            CFRelease(chroma);
            double took = done.GPUEndTime - done.GPUStartTime;
            @synchronized (self->_pending) { self->_presentSum += took; self->_presentCount++; }
        }];
        [commands commit];
    }
}

- (double)refreshAtOrAfter:(double)time {
    double period, grid;
    @synchronized (_pending) { period = _period; grid = _slotVsync; }
    return grid + ceil((time - grid) / period - 1e-6) * period;
}

- (void)showRest {
    if (!_restFrame) return;
    double target = MAX(_lastTarget + _period, [self refreshAtOrAfter:HostNow() + kCommitLead]);
    [self show:_restNative target:target];
    _lastTarget = target;
    CVPixelBufferRelease(_previousFrame);
    _previousFrame = _restFrame;
    _previousTime = _restTime;
    CVPixelBufferRelease(_previousNative);
    _previousNative = (_dumpDirectory || _flow) ? CVPixelBufferRetain(_restNative) : NULL;
    CVPixelBufferRelease(_previousFlowInput);
    _previousFlowInput = NULL;
    _restFrame = NULL;
    CVPixelBufferRelease(_restNative);
    _restNative = NULL;
}

- (void)handle:(CVPixelBufferRef)native small:(CVPixelBufferRef)small input:(CVPixelBufferRef)input received:(double)received {
    if (_statsStart == 0) _statsStart = received;
    _received++;
    NSUInteger generation = ++_generation;
    CVPixelBufferRelease(_restFrame);
    _restFrame = NULL;
    CVPixelBufferRelease(_restNative);
    _restNative = NULL;

    if ([self isDuplicate:small received:received]) {
        _restFrame = CVPixelBufferRetain(input);
        _restNative = CVPixelBufferRetain(native);
        _restTime = received;
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.06 * NSEC_PER_SEC)), _queue, ^{
            if (self->_generation == generation) [self showRest];
        });
    } else {
        _unique++;
        if (_sceneCut) _cuts++;
        double gap = _previousFrame ? received - _previousTime : 0;
        if (gap > 0.025 && gap < 0.045) _sourceInterval += 0.1 * (gap - _sourceInterval);
        double period;
        @synchronized (_pending) { period = _period; }
        double step = MAX(1.0, round(_sourceInterval / 2 / period)) * period;

        CVPixelBufferRef middleOut = NULL;
        if (_previousFrame && gap > 1.5 * step && gap < 0.100 && !_sceneCut) {
            double begin = HostNow();
            if (_flow && _previousNative) middleOut = [self flowMiddleFrom:_previousNative at:_previousTime to:native at:received];
            CVPixelBufferRef middle = middleOut ? NULL : [self interpolateFrom:_previousFrame at:_previousTime to:input at:received];
            _interpSum += HostNow() - begin;
            if (middleOut) {
                _interpolated++;
                if (_dumpDirectory && _lastMean > 6.0 && _dumpCount < 15 && _dumpCandidates++ % 10 == 0) {
                    [self dumpPrevious:_previousNative middle:middleOut current:native nativePrevious:NULL nativeCurrent:NULL];
                }
            }
            if (middle && _dumpDirectory && _lastMean > 6.0 && _dumpCount < 15 && _dumpCandidates++ % 10 == 0) {
                [self dumpPrevious:_previousFrame middle:middle current:input nativePrevious:_previousNative nativeCurrent:native];
            }
            if (middle) {
                _interpolated++;
                BOOL large = CVPixelBufferGetWidth(native) == (size_t)kProcessWidth * 3 / 2 && CVPixelBufferGetHeight(native) == (size_t)kProcessHeight * 3 / 2;
                middleOut = [self finished:middle large:large];
                CVPixelBufferRelease(middle);
            }
        }

        double afterPrevious = _lastTarget + step;
        double earliest = [self refreshAtOrAfter:HostNow() + kCommitLead];
        if (afterPrevious > earliest + 2 * step + period / 2) afterPrevious = earliest;
        double first = MAX(MAX(afterPrevious, earliest), [self refreshAtOrAfter:received] + 2 * step);
        if (middleOut && first - received > 3 * step + period / 2) {
            CVPixelBufferRelease(middleOut);
            middleOut = NULL;
            _skipped++;
        }
        if (_lastTarget > 0 && gap < 0.100 && first > afterPrevious + period / 2) _holds += (int)llround((first - afterPrevious) / step);
        if (middleOut) {
            [self show:middleOut target:first];
            CVPixelBufferRelease(middleOut);
            first += step;
        }
        first = MAX(first, [self refreshAtOrAfter:HostNow() + kCommitLead]);
        [self show:native target:first];
        _lastTarget = first;
        _delaySum += first - received;
        _delayCount++;
        CVPixelBufferRelease(_previousFrame);
        _previousFrame = CVPixelBufferRetain(input);
        _previousTime = received;
        CVPixelBufferRelease(_previousNative);
        _previousNative = (_dumpDirectory || _flow) ? CVPixelBufferRetain(native) : NULL;
        CVPixelBufferRelease(_previousFlowInput);
        _previousFlowInput = _currentFlowInput;
        _currentFlowInput = NULL;
    }

    double elapsed = received - _statsStart;
    if (elapsed < 2.0) return;
    if (_debug) {
        int shown, missed, dropped, shifts, resynced, glass[4];
        double present;
        @synchronized (_pending) {
            shown = _shown;
            missed = _missed;
            dropped = _dropped;
            shifts = _shifts;
            resynced = _resynced;
            memcpy(glass, _glass, sizeof(glass));
            memset(_glass, 0, sizeof(_glass));
            present = _presentCount ? _presentSum / _presentCount : 0;
            _shown = _missed = _dropped = _shifts = _resynced = _presentCount = 0;
            _presentSum = 0;
        }
        fprintf(stderr, "smooth: in %.1f fps (unique %.1f) | out %.1f fps | held %d, skipped %d, missed %d, cuts %d | "
                "on glass +0/+1/+2/+3 %d/%d/%d/%d, shifts %d, dropped %d, resynced %d | "
                "interp %.1f ms, upscale %.1f ms, present %.1f ms GPU | delay %.1f ms\n",
                _received / elapsed, _unique / elapsed, shown / elapsed, _holds, _skipped, missed, _cuts,
                glass[0], glass[1], glass[2], glass[3], shifts, dropped, resynced,
                _interpolated ? _interpSum / _interpolated * 1000 : 0, _scaled ? _scaleSum / _scaled * 1000 : 0, present * 1000,
                _delayCount ? _delaySum / _delayCount * 1000 : 0);
    } else {
        @synchronized (_pending) { _shown = _missed = _dropped = _shifts = _resynced = _presentCount = 0; _presentSum = 0; memset(_glass, 0, sizeof(_glass)); }
    }
    _statsStart = received;
    _received = _unique = _interpolated = _scaled = _cuts = _holds = _skipped = _delayCount = 0;
    _interpSum = _scaleSum = _delaySum = 0;
}

- (void)submit:(CVPixelBufferRef)frame {
    double received = HostNow();
    CVPixelBufferRetain(frame);
    if (!_attached) {
        _attached = YES;
        dispatch_async(dispatch_get_main_queue(), ^{ [self attach]; });
    }
    dispatch_async(dispatch_get_main_queue(), ^{ [self unhide]; });
    dispatch_async(_queue, ^{
        [self prepare];
        OSType format = CVPixelBufferGetPixelFormatType(frame);
        BOOL showable = format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
        BOOL fits = format == kProcessFormat && CVPixelBufferGetWidth(frame) == (size_t)kProcessWidth && CVPixelBufferGetHeight(frame) == (size_t)kProcessHeight;
        CVPixelBufferRef small = NULL;
        if (fits) {
            small = CVPixelBufferRetain(frame);
        } else {
            CVPixelBufferPoolCreatePixelBuffer(NULL, self->_smallPool, &small);
            if (small && VTPixelTransferSessionTransferImage(self->_transfer, frame, small) != noErr) { CVPixelBufferRelease(small); small = NULL; }
        }
        CVPixelBufferRef full = NULL;
        if (small && self->_fullPool) {
            CVPixelBufferPoolCreatePixelBuffer(NULL, self->_fullPool, &full);
            if (full) CVBufferPropagateAttachments(frame, full);
            if (full && VTPixelTransferSessionTransferImage(self->_transfer, frame, full) != noErr) { CVPixelBufferRelease(full); full = NULL; }
        }
        if (small) [self handle:(showable ? frame : small) small:small input:(full ?: small) received:received];
        CVPixelBufferRelease(full);
        CVPixelBufferRelease(small);
        CVPixelBufferRelease(frame);
    });
}
@end

static Smoother *Shared(void) {
    static Smoother *smoother;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ smoother = [Smoother new]; });
    return smoother;
}

void SmoothSubmit(CVPixelBufferRef frame) {
    [Shared() submit:frame];
}

void SmoothHide(void) {
    dispatch_async(dispatch_get_main_queue(), ^{ [Shared() hide]; });
}
