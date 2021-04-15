### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 249249ca-9c7e-11eb-3208-49fe9067af66
begin
	using Pkg
	Pkg.status()
	Pkg.activate("Project.toml")
end

# ╔═╡ 74940111-21ef-4feb-a4cb-c294fee976c5
begin
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 792c8d39-5d40-4bd2-8f6c-89ca2dca06a1
begin
	using Flux
	NN = Chain(x -> [x], 
	           Dense(1,32,tanh),
	           Dense(32,1),
	           first) 
end

# ╔═╡ b18ee325-d314-4060-a372-aa5f439a48fc
using Statistics

# ╔═╡ f69f6a88-a841-462d-b102-41feac9f89e1
md"# Neural network learning differential equation"

# ╔═╡ 0090a0ab-a229-42d9-a1d6-402b4b894f6c
md" ## Differential Equation"

# ╔═╡ fc21583b-0c34-4176-baef-128cc5c1f38b
md" Let's say we want to our neural network to learn the solution of a differential equation. 

$$u' = x^2 + cos(x)$$ with $u(0) = 1$."

# ╔═╡ 0efcec63-7b25-4ed7-856f-8b50853d1588
f(x) = cos(2π*x) + x^2

# ╔═╡ bc6db2be-076f-428f-812e-1118a7928e94
md"We can approximate the solution of the equation $u(x)$ with a neural network"

# ╔═╡ ed04bec1-1ea4-450c-95dd-706f5ac662ff
md"### Neural Network"

# ╔═╡ 96083ecd-74f3-4dad-960e-ac89efacef38
md" We can encode the initial vaue in by considering $u(x)$ as following."

# ╔═╡ 66a7a91a-c791-4105-829d-add50f8ad0cf
 u(x) = x*NN(x) + 1

# ╔═╡ 55e4fd70-59f1-4313-a273-c4b297dfd99e
begin
	using Plots
	t = 0:0.001:1.0
	plot(t,u.(t),label="NN")
	plot!(t,1.0 .+ sin.(2π.*t)/2π + (t.^3)/3 , label = "True Solution")
end

# ╔═╡ 1eed2240-2b57-4c02-9dd1-1586ad6bc8f3
ϵ = sqrt(eps(Float32))

# ╔═╡ 7d7279a0-c36b-4a3c-b337-26d3ca9f13df
md" Let's calculate the derivative. Note that we can also use gradient function using Automatic Differentiation to do this step."

# ╔═╡ 8ffb9628-a2ba-4b63-bafb-483d990298a7
du(x) = (u(x+ϵ)-u(x))/ϵ

# ╔═╡ 21830e01-ca56-4359-ad06-23a099e2d416
loss() = mean(abs2(du(x) - f(x)) for x in 0:1f-2:1f0)

# ╔═╡ cf3dbb64-f225-4623-bb03-d9c5d56703b8
Flux.train!(loss, Flux.params(NN), Iterators.repeated((), 5000), Flux.Descent(0.01))

# ╔═╡ 992dc86a-f43a-4802-abad-ed6df58a02e0
md" This equation can be solved analytically. Let's compare how is approximation doing compared to the exact solution."

# ╔═╡ 816b2a29-08a6-4462-82c4-ca43fcb94765
md"## Another Equation"

# ╔═╡ 3ca73a93-cacf-4217-b9ff-d6cb7fa07aba
md" Let us consider another equation

$$y' + 2y = 3e^t$$
with initial condition $y(0) = 3$, and solution $ y = 2e^{-2t} + e^t$"

# ╔═╡ 5ab6f26c-a829-4774-af57-db4c5a4f0fc8
begin
	NN2 = Chain(t -> [t], 
	           Dense(1,32,tanh),
	           Dense(32,1),
	           first) 
	
	
	g(t) = t*NN2(t) + 3f0
	#using Statistics
	#ϵ = sqrt(eps(Float32))
	loss2() = mean(abs2(((g(t+ϵ)-g(t))/ϵ) - (3ℯ^t - 2g(t))) for t in 0:1f-2:1f0)
	loss2()
	data = Iterators.repeated((), 5000)
	Flux.train!(loss2, Flux.params(NN2), data,ADAM(0.1))
	
	#using Plots
	#t = 0:0.001:1.0
	plot(t,g.(t),label="NN")
	plot!(t,2ℯ.^(-2t) + ℯ.^t, label = "True Solution")
end

# ╔═╡ Cell order:
# ╟─249249ca-9c7e-11eb-3208-49fe9067af66
# ╟─74940111-21ef-4feb-a4cb-c294fee976c5
# ╟─f69f6a88-a841-462d-b102-41feac9f89e1
# ╟─0090a0ab-a229-42d9-a1d6-402b4b894f6c
# ╟─fc21583b-0c34-4176-baef-128cc5c1f38b
# ╠═0efcec63-7b25-4ed7-856f-8b50853d1588
# ╟─bc6db2be-076f-428f-812e-1118a7928e94
# ╟─ed04bec1-1ea4-450c-95dd-706f5ac662ff
# ╠═792c8d39-5d40-4bd2-8f6c-89ca2dca06a1
# ╟─96083ecd-74f3-4dad-960e-ac89efacef38
# ╠═66a7a91a-c791-4105-829d-add50f8ad0cf
# ╠═1eed2240-2b57-4c02-9dd1-1586ad6bc8f3
# ╟─7d7279a0-c36b-4a3c-b337-26d3ca9f13df
# ╠═8ffb9628-a2ba-4b63-bafb-483d990298a7
# ╠═b18ee325-d314-4060-a372-aa5f439a48fc
# ╠═21830e01-ca56-4359-ad06-23a099e2d416
# ╠═cf3dbb64-f225-4623-bb03-d9c5d56703b8
# ╟─992dc86a-f43a-4802-abad-ed6df58a02e0
# ╠═55e4fd70-59f1-4313-a273-c4b297dfd99e
# ╟─816b2a29-08a6-4462-82c4-ca43fcb94765
# ╟─3ca73a93-cacf-4217-b9ff-d6cb7fa07aba
# ╠═5ab6f26c-a829-4774-af57-db4c5a4f0fc8
