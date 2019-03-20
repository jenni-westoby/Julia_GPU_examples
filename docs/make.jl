using Documenter

module MyModule
end

makedocs(modules=[MyModule],
build = "build",
repo = "github.com/jenni-westoby/Julia_GPU_examples.git",
sitename = "Learn to Develop GPU Software with Julia",
pages = Any[
"Home" => "index.md",
"Getting Started" => Any["GPU_background.md","Accessing_GPUs.md",
"Setup_Julia.md"],
"Developing With GPUs" => Any["Vector_addition.md", "Vector_dot_product.md",
"Streaming.md", "Challenges.md"],
"Performance" => Any["Profiling.md", "Performance_thoughts.md"],
"Further Reading" => "Further_Reading.md",
"About the Author" => "About_the_author.md"])

deploydocs(repo = "github.com/jenni-westoby/Julia_GPU_examples.git")
