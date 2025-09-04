#import "ModelRunner.h"

@interface ModelRunner ()
@property(nonatomic, strong) MLModel *model;
@end

@implementation ModelRunner

- (instancetype)initWithModel:(MLModel *)model {
    if ((self = [super init])) {
        _model = model;
    }
    return self;
}

- (NSArray<NSNumber *> *)predictWithInput:(NSArray<NSNumber *> *)input {
    NSError *error = nil;
    MLMultiArray *tokens = [[MLMultiArray alloc] initWithShape:@[@1, @(input.count)]
                                                      dataType:MLMultiArrayDataTypeInt32
                                                         error:&error];
    if (error) {
        return @[];
    }
    for (NSUInteger i = 0; i < input.count; ++i) {
        tokens[i] = input[i];
    }
    NSDictionary *features = @{ @"tokens" : tokens };
    id<MLFeatureProvider> provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
    if (error) {
        return @[];
    }
    id<MLFeatureProvider> result = [self.model predictionFromFeatures:provider error:&error];
    if (error) {
        return @[];
    }
    MLMultiArray *output = [result featureValueForName:@"logits"].multiArrayValue;
    NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:output.count];
    for (NSInteger i = 0; i < output.count; i++) {
        values[i] = output[i];
    }
    return values;
}

@end
