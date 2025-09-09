import Foundation
#if canImport(Tokenizers)
import Tokenizers
#endif

/// Hugging Face tokenizer for the EXAONE model loaded at runtime.
class EXAONETokenizer {
    private let tokenizer: AnyObject
    
    // Common token IDs (these should be configured based on the actual tokenizer)
    let eosTokenId: Int = 2  // Common EOS token ID, should be updated based on actual tokenizer
    let padTokenId: Int = 0  // Common PAD token ID

    init() {
        #if canImport(Tokenizers)
        // Download tokenizer.json from Hugging Face once at startup.
        let url = URL(string: "https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/resolve/main/tokenizer.json")!
        do {
            let data = try Data(contentsOf: url)
            tokenizer = try Tokenizer.fromBuffer(data) as AnyObject
            print("EXAONE tokenizer loaded successfully")
        } catch {
            print("Error loading tokenizer: \(error)")
            // Fallback to dummy tokenizer
            tokenizer = NSObject()
        }
        #else
        print("Tokenizers framework not available, using dummy tokenizer")
        tokenizer = NSObject()
        #endif
    }

    func encode(text: String) -> [NSNumber] {
        #if canImport(Tokenizers)
        guard let tok = tokenizer as? Tokenizer else {
            // Fallback encoding for development/testing
            return text.utf8.map { NSNumber(value: Int($0)) }
        }
        
        do {
            let encoding = try tok.encode(text: text)
            return encoding.ids.map { NSNumber(value: Int($0)) }
        } catch {
            print("Error encoding text: \(error)")
            return []
        }
        #else
        // Simple fallback for when Tokenizers framework is not available
        return text.utf8.map { NSNumber(value: Int($0)) }
        #endif
    }

    func decode(tokens: [Int]) -> String {
        #if canImport(Tokenizers)
        guard let tok = tokenizer as? Tokenizer else {
            // Fallback decoding
            return String(bytes: tokens.compactMap { UInt8(exactly: $0) }, encoding: .utf8) ?? ""
        }
        
        do {
            return try tok.decode(tokens: tokens.map { UInt32($0) })
        } catch {
            print("Error decoding tokens: \(error)")
            return ""
        }
        #else
        // Simple fallback
        return String(bytes: tokens.compactMap { UInt8(exactly: $0) }, encoding: .utf8) ?? ""
        #endif
    }
}

