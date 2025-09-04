#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface ModelRunner : NSObject
- (instancetype)initWithModel:(MLModel *)model;
- (NSArray<NSNumber *> *)predictWithInput:(NSArray<NSNumber *> *)input;
@end

NS_ASSUME_NONNULL_END
