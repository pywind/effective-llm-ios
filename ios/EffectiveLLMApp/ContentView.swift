import SwiftUI

struct ContentView: View {
    @State private var input: String = ""
    @State private var output: String = ""
    private let model = LLMModel()

    var body: some View {
        VStack {
            TextEditor(text: $input)
                .frame(height: 120)
                .border(Color.gray)
                .padding()
            Button("Generate") {
                if let result = model?.generate(text: input) {
                    output = result
                }
            }
            .padding()
            TextEditor(text: $output)
                .frame(height: 120)
                .border(Color.gray)
                .padding()
        }
        .padding()
    }
}
