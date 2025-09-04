import Foundation

/// A toy tokenizer using Unicode scalar values.
class SimpleTokenizer {
    func encode(text: String) -> [NSNumber] {
        text.unicodeScalars.map { NSNumber(value: Int($0.value)) }
    }

    func decode(tokens: [Int]) -> String {
        let scalars = tokens.compactMap { UnicodeScalar($0) }
        return String(String.UnicodeScalarView(scalars))
    }
}
