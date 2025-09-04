import Foundation
#if canImport(Tokenizers)
import Tokenizers
#endif

/// Hugging Face tokenizer for the EXAONE model loaded at runtime.
class EXAONETokenizer {
    private let tokenizer: AnyObject

    init() {
        #if canImport(Tokenizers)
        // Download tokenizer.json from Hugging Face once at startup.
        let url = URL(string: "https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/resolve/main/tokenizer.json")!
        let data = try! Data(contentsOf: url)
        tokenizer = try! Tokenizer.fromBuffer(data) as AnyObject
        #else
        tokenizer = NSObject()
        #endif
    }

    func encode(text: String) -> [NSNumber] {
        #if canImport(Tokenizers)
        let encoding = try! (tokenizer as! Tokenizer).encode(text: text)
        return encoding.ids.map { NSNumber(value: Int($0)) }
        #else
        return []
        #endif
    }

    func decode(tokens: [Int]) -> String {
        #if canImport(Tokenizers)
        return (try? (tokenizer as! Tokenizer).decode(tokens: tokens.map { UInt32($0) })) ?? ""
        #else
        return ""
        #endif
    }
}

