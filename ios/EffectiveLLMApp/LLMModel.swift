import Foundation
import CoreML

/// Swift wrapper around the CoreML model using the Objective-C runner.
final class LLMModel {
    private let runner: ModelRunner
    private let tokenizer = EXAONETokenizer()
    
    // Generation parameters
    private let maxNewTokens: Int = 50
    private let temperature: Float = 0.7

    init?() {
        guard let url = Bundle.main.url(forResource: "Model", withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: url) else {
            print("Failed to load CoreML model")
            return nil
        }
        runner = ModelRunner(model: model)
        print("LLM Model initialized successfully")
    }

    func generate(text: String) -> String {
        // Reset cache for new generation
        runner.resetCache()
        
        let tokens = tokenizer.encode(text: text)
        guard !tokens.isEmpty else {
            return "Error: Failed to tokenize input"
        }
        
        var generatedTokens = tokens.map { $0.intValue }
        
        // Generate tokens one by one
        for step in 0..<maxNewTokens {
            // Use the last token for prediction (or all tokens for first step)
            let inputTokens = step == 0 ? tokens : [NSNumber(value: generatedTokens.last!)]
            
            let logits = runner.predictWithInput(inputTokens, resetCache: false)
            guard !logits.isEmpty else {
                print("Error: Empty logits returned")
                break
            }
            
            // Apply temperature sampling
            let nextTokenId = sampleFromLogits(logits, temperature: temperature)
            generatedTokens.append(nextTokenId)
            
            // Check for EOS token
            if nextTokenId == tokenizer.eosTokenId {
                break
            }
        }
        
        return tokenizer.decode(tokens: generatedTokens)
    }
    
    private func sampleFromLogits(_ logits: [NSNumber], temperature: Float) -> Int {
        // Convert to float array
        let floatLogits = logits.map { $0.floatValue }
        
        // Apply temperature
        let scaledLogits = floatLogits.map { $0 / temperature }
        
        // Find max for numerical stability
        let maxLogit = scaledLogits.max() ?? 0.0
        
        // Compute softmax probabilities
        let expLogits = scaledLogits.map { exp($0 - maxLogit) }
        let sumExp = expLogits.reduce(0, +)
        let probabilities = expLogits.map { $0 / sumExp }
        
        // Sample from multinomial distribution
        let randomValue = Float.random(in: 0..<1)
        var cumulativeProb: Float = 0.0
        
        for (index, prob) in probabilities.enumerated() {
            cumulativeProb += prob
            if randomValue <= cumulativeProb {
                return index
            }
        }
        
        // Fallback to last token if something goes wrong
        return probabilities.count - 1
    }
}
