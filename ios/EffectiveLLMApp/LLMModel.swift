import Foundation
import CoreML

/// Swift wrapper around the CoreML model using the Objective-C runner.
final class LLMModel {
    private let runner: ModelRunner
    private let tokenizer = EXAONETokenizer()

    init?() {
        guard let url = Bundle.main.url(forResource: "Model", withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: url) else {
            return nil
        }
        runner = ModelRunner(model: model)
    }

    func generate(text: String) -> String {
        let tokens = tokenizer.encode(text: text)
        let logits = runner.predictWithInput(tokens)
        let ids = logits.map { $0.intValue }
        return tokenizer.decode(tokens: ids)
    }
}
