#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface ModelRunner : NSObject
- (instancetype)initWithModel:(MLModel *)model;
- (NSArray<NSNumber *> *)predictWithInput:(NSArray<NSNumber *> *)input;
- (NSArray<NSNumber *> *)predictWithInput:(NSArray<NSNumber *> *)input resetCache:(BOOL)resetCache;
- (void)resetCache;
@end

NS_ASSUME_NONNULL_END
