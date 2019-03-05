using Documenter

module MyModule
end

makedocs(modules=[MyModule],
build = "build",
repo = "https://github.com/jenni-westoby/Julia_GPU_examples.git",
pages = Any[
"Home" => "index.md",
"Getting Started" => Any["Accessing_GPUs.md"]])

deploydocs(deps   = Deps.pip("mkdocs", "python-markdown-math"),
repo = "https://github.com/jenni-westoby/Julia_GPU_examples.git",
julia  = "1.0.3",
osname = "linux")
