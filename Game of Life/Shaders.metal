//
//  Shaders.metal
//  Game of Life
//
//  Created by David Nadoba on 11.11.18.
//  Copyright © 2018 David Nadoba. All rights reserved.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
} ColorInOut;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]])
{
    ColorInOut out;

    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
    out.texCoord = in.texCoord;

    return out;
}

const int2 wrap2d(const int2 index, const int2 size) {
    return abs(index) % size;
}
const int index2d(const int2 index, const int2 size) {
    return index.y * size.y + index.x;
}
const int indexWrap2d(const int2 index, const int2 size) {
    return index2d(wrap2d(index, size), size);
}

//const uint8_t countAliveNeighbours(const int2 index) {
//    return 0;
//}

const uint8_t isAliveInNextGen(const uint8_t isCenterAlive, const uint8_t aliveNeighbours) {
    if (isCenterAlive) {
        if (aliveNeighbours < 2 || aliveNeighbours > 3) {
            return 0;
        }
        return 1;
        // dead
    } else {
        if (aliveNeighbours == 3) {
            return 1;
        }
        return 0;
    }
}

constant int width = 1024;
constant int height = 1024;

kernel void gameOfLifeKernel(constant uint8_t * inField  [[ buffer(BufferIndexInputField) ]],
                             device uint8_t * outField  [[ buffer(BufferIndexOutputField) ]],
                             uint2 position [[thread_position_in_grid]])
{
    int2 index = int2(position);
    int2 size = int2(width, height);
    
    uint8_t aliveCount = 0;
    aliveCount +=       inField[indexWrap2d(index + int2(-1, -1), size)];
    aliveCount +=       inField[indexWrap2d(index + int2(-1,  0), size)];
    aliveCount +=       inField[indexWrap2d(index + int2(-1, +1), size)];
    
    aliveCount +=       inField[indexWrap2d(index + int2( 0, -1), size)];
    uint8_t isAllive =  inField[indexWrap2d(index + int2( 0,  0), size)];
    aliveCount +=       inField[indexWrap2d(index + int2( 0, +1), size)];
    
    aliveCount +=       inField[indexWrap2d(index + int2(+1, -1), size)];
    aliveCount +=       inField[indexWrap2d(index + int2(+1,  0), size)];
    aliveCount +=       inField[indexWrap2d(index + int2(+1, +1), size)];


    outField[index2d(index, size)] = isAliveInNextGen(isAllive, aliveCount);
}

fragment float4 fragmentShader(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               texture2d<half> colorMap     [[ texture(TextureIndexColor) ]],
                               constant uint8_t * field [[ buffer(BufferIndexField) ]])
{
    
    float2 size = float2(width, height);
    
    int2 position = int2(in.texCoord * size);
    int index = position.y * size.y + position.x;
    uint8_t element = field[index];
    return float4(element, 0, 0, 0);
    
//    half4 colorSample   = colorMap.sample(colorSampler, in.texCoord.xy);
//
//    return float4(colorSample);
}
