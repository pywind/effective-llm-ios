#import "ModelRunner.h"

@interface ModelRunner ()
@property(nonatomic, strong) MLModel *model;
@property(nonatomic, strong) MLMultiArray *keyCache;
@property(nonatomic, strong) MLMultiArray *valueCache;
@property(nonatomic, strong) MLPredictionOptions *predictionOptions;
@end

@implementation ModelRunner

- (instancetype)initWithModel:(MLModel *)model {
    if ((self = [super init])) {
        _model = model;
        [self setupCacheAndOptions];
    }
    return self;
}

- (void)setupCacheAndOptions {
    // Initialize KV cache with proper shapes
    // This should match the cache shape from the conversion script
    NSError *error = nil;
    
    // Get the expected cache shape from model description
    MLFeatureDescription *keyCacheDesc = nil;
    MLFeatureDescription *valueCacheDesc = nil;
    
    for (MLFeatureDescription *inputDesc in self.model.modelDescription.inputDescriptionsByName.allValues) {
        if ([inputDesc.name isEqualToString:@"key_cache"]) {
            keyCacheDesc = inputDesc;
        } else if ([inputDesc.name isEqualToString:@"value_cache"]) {
            valueCacheDesc = inputDesc;
        }
    }
    
    if (keyCacheDesc && keyCacheDesc.multiArrayConstraint) {
        NSArray *shape = keyCacheDesc.multiArrayConstraint.shape;
        self.keyCache = [[MLMultiArray alloc] initWithShape:shape
                                                   dataType:MLMultiArrayDataTypeFloat16
                                                      error:&error];
        if (error) {
            NSLog(@"Error creating key cache: %@", error.localizedDescription);
        } else {
            // Initialize with zeros
            for (NSInteger i = 0; i < self.keyCache.count; i++) {
                self.keyCache[i] = @0.0f;
            }
        }
    }
    
    if (valueCacheDesc && valueCacheDesc.multiArrayConstraint) {
        NSArray *shape = valueCacheDesc.multiArrayConstraint.shape;
        self.valueCache = [[MLMultiArray alloc] initWithShape:shape
                                                     dataType:MLMultiArrayDataTypeFloat16
                                                        error:&error];
        if (error) {
            NSLog(@"Error creating value cache: %@", error.localizedDescription);
        } else {
            // Initialize with zeros
            for (NSInteger i = 0; i < self.valueCache.count; i++) {
                self.valueCache[i] = @0.0f;
            }
        }
    }
    
    // Setup prediction options for better performance
    self.predictionOptions = [[MLPredictionOptions alloc] init];
    self.predictionOptions.usesCPUOnly = NO;  // Allow Neural Engine usage
}

- (void)resetCache {
    // Reset the KV cache to zeros
    if (self.keyCache) {
        for (NSInteger i = 0; i < self.keyCache.count; i++) {
            self.keyCache[i] = @0.0f;
        }
    }
    if (self.valueCache) {
        for (NSInteger i = 0; i < self.valueCache.count; i++) {
            self.valueCache[i] = @0.0f;
        }
    }
}

- (NSArray<NSNumber *> *)predictWithInput:(NSArray<NSNumber *> *)input {
    return [self predictWithInput:input resetCache:NO];
}

- (NSArray<NSNumber *> *)predictWithInput:(NSArray<NSNumber *> *)input resetCache:(BOOL)resetCache {
    NSError *error = nil;
    
    if (resetCache) {
        [self resetCache];
    }
    
    // Create input tensor
    MLMultiArray *tokens = [[MLMultiArray alloc] initWithShape:@[@1, @(input.count)]
                                                      dataType:MLMultiArrayDataTypeInt32
                                                         error:&error];
    if (error) {
        NSLog(@"Error creating input tokens: %@", error.localizedDescription);
        return @[];
    }
    
    for (NSUInteger i = 0; i < input.count; ++i) {
        tokens[i] = input[i];
    }
    
    // Create feature provider with all inputs including KV cache
    NSMutableDictionary *features = [@{
        @"tokens": tokens
    } mutableCopy];
    
    if (self.keyCache) {
        features[@"key_cache"] = self.keyCache;
    }
    if (self.valueCache) {
        features[@"value_cache"] = self.valueCache;
    }
    
    id<MLFeatureProvider> provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
    if (error) {
        NSLog(@"Error creating feature provider: %@", error.localizedDescription);
        return @[];
    }
    
    // Perform prediction
    id<MLFeatureProvider> result = [self.model predictionFromFeatures:provider 
                                                               options:self.predictionOptions 
                                                                 error:&error];
    if (error) {
        NSLog(@"Error during prediction: %@", error.localizedDescription);
        return @[];
    }
    
    // Update KV cache with outputs
    MLFeatureValue *keyCacheOut = [result featureValueForName:@"key_cache_out"];
    if (keyCacheOut && keyCacheOut.multiArrayValue) {
        self.keyCache = keyCacheOut.multiArrayValue;
    }
    
    MLFeatureValue *valueCacheOut = [result featureValueForName:@"value_cache_out"];
    if (valueCacheOut && valueCacheOut.multiArrayValue) {
        self.valueCache = valueCacheOut.multiArrayValue;
    }
    
    // Extract logits
    MLMultiArray *output = [result featureValueForName:@"logits"].multiArrayValue;
    if (!output) {
        NSLog(@"No logits output found");
        return @[];
    }
    
    NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:output.count];
    for (NSInteger i = 0; i < output.count; i++) {
        values[i] = output[i];
    }
    return values;
}

@end
