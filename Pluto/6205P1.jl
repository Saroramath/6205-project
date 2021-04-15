### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ de0d3c88-78fc-41c4-af86-cacb0b597ae2
begin
	using Pkg
	Pkg.status()
	Pkg.activate("Project.toml")
end

# ╔═╡ 09cf94d2-9c72-11eb-1244-055426767c8f
begin
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 5510a4aa-dd5e-48d4-a467-a15d33c4f65e
using Flux

# ╔═╡ 85826223-0e3c-4037-b732-77696b992122
using Plots

# ╔═╡ c02425f0-54ca-4992-a331-2f9535a9317e
md"# Neural Network"

# ╔═╡ 8792dc02-46b2-459a-92c1-236b736ba014
md"## Mathematical functions"

# ╔═╡ b0eef678-7db1-445e-8ac3-53b7ea1bd84a
md"Let us consider any function, say $y = \sin(x)$."

# ╔═╡ 7d13fc6e-d05a-4952-a8a6-5e59f29b6e41
x = sort(rand(100))

# ╔═╡ 659e630b-24bb-4a2b-b55b-02389db9f0a4
y = sin.(2π.*x)

# ╔═╡ fcc1e8a1-a536-4724-8019-9bd5402ba870
plot(x,y)

# ╔═╡ d6cfebf1-f89a-43c2-abcc-2825e3ad03e0
md"## Neural Network"

# ╔═╡ 2ada5d9d-2be3-401e-b0fd-dc220134ee8e
md" The network can be defined s a cobination of layers. Each layer in Flux is a function. The chain method is equivalent to composition of functions."

# ╔═╡ 07b05516-cef1-4593-a7d3-3a272b7c7268
NN = Chain(x -> [x], Dense(1,32,tanh), Dense(32,32,tanh),Dense(32,1),first)

# ╔═╡ 7368f0ae-0d72-4d50-835a-8668053348dc
pred = NN.(x)

# ╔═╡ e0c0e3bd-cd46-42fe-a8a2-5219335c7547
typeof(pred)

# ╔═╡ 6d13ccbf-80ac-4147-a3bc-9dfd80018503
md"## Neural Network as a function approximator"

# ╔═╡ 39031df5-9ebf-403f-8128-786a1693227c
md" Let's see if the network can approximate the $\sin$ function."

# ╔═╡ c063963a-f898-46cf-8b53-216def9bd395
loss() = sum(abs2,NN.(x).-y)

# ╔═╡ 7183254e-14e8-454e-bb53-868bf487bbed
#loss() = Flux.mse(NN.(x),y)

# ╔═╡ 9fd8c843-ba03-4a01-bcc9-cfcc28db4474
Flux.train!(loss, params(NN),Iterators.repeated((), 1000) ,ADAM(0.1))

# ╔═╡ 320733a7-cf22-4864-a651-9e73022cd638
loss()

# ╔═╡ 7eee2bf7-9714-4ff8-ab99-ccafbee77df1
plot(x,NN.(x))

# ╔═╡ d1746bf0-372c-44ea-aead-eb62278d8abe
md"It will only learn what is taught"

# ╔═╡ 3140a6c2-3861-468a-8a45-4a911b21a4e6
t = 0:0.01:2

# ╔═╡ b4e7cf99-a6fe-4268-afda-764311deff2d
plot(t,NN.(t))

# ╔═╡ Cell order:
# ╟─de0d3c88-78fc-41c4-af86-cacb0b597ae2
# ╟─09cf94d2-9c72-11eb-1244-055426767c8f
# ╟─c02425f0-54ca-4992-a331-2f9535a9317e
# ╟─5510a4aa-dd5e-48d4-a467-a15d33c4f65e
# ╟─85826223-0e3c-4037-b732-77696b992122
# ╟─8792dc02-46b2-459a-92c1-236b736ba014
# ╟─b0eef678-7db1-445e-8ac3-53b7ea1bd84a
# ╠═7d13fc6e-d05a-4952-a8a6-5e59f29b6e41
# ╠═659e630b-24bb-4a2b-b55b-02389db9f0a4
# ╠═fcc1e8a1-a536-4724-8019-9bd5402ba870
# ╟─d6cfebf1-f89a-43c2-abcc-2825e3ad03e0
# ╟─2ada5d9d-2be3-401e-b0fd-dc220134ee8e
# ╠═07b05516-cef1-4593-a7d3-3a272b7c7268
# ╠═7368f0ae-0d72-4d50-835a-8668053348dc
# ╠═e0c0e3bd-cd46-42fe-a8a2-5219335c7547
# ╟─6d13ccbf-80ac-4147-a3bc-9dfd80018503
# ╟─39031df5-9ebf-403f-8128-786a1693227c
# ╠═c063963a-f898-46cf-8b53-216def9bd395
# ╠═7183254e-14e8-454e-bb53-868bf487bbed
# ╠═9fd8c843-ba03-4a01-bcc9-cfcc28db4474
# ╠═320733a7-cf22-4864-a651-9e73022cd638
# ╠═7eee2bf7-9714-4ff8-ab99-ccafbee77df1
# ╟─d1746bf0-372c-44ea-aead-eb62278d8abe
# ╠═3140a6c2-3861-468a-8a45-4a911b21a4e6
# ╠═b4e7cf99-a6fe-4268-afda-764311deff2d
