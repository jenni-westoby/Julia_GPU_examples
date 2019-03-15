using Documenter

module MyModule
end

makedocs(modules=[MyModule],
build = "build",
repo = "github.com/jenni-westoby/Julia_GPU_examples.jl.git",
sitename = "Julia_GPU_examples.jl",
pages = Any[
"Home" => "index.md",
"Getting Started" => Any["GPU_background.md","Accessing_GPUs.md",
"Setup_Julia.md"],
"Developing With GPUs" => Any["Vector_addition.md", "Vector_dot_product.md",
"Streaming.md"],
"Performance" => Any["Profiling.md", "Performance_thoughts.md"],
"Summary" => "Summary.md",
"Further Reading" => "Further_Reading.md",
"About the Author" => "About_the_author.md"])

deploydocs(deps   = nothing,
repo = "github.com/jenni-westoby/Julia_GPU_examples.jl.git",
julia  = "1.0.3",
osname = "linux")
