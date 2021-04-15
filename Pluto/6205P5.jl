### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ e0eb73a2-9d3a-11eb-3d4f-77229772a63b
begin
	using Pkg
	Pkg.status()
	Pkg.activate("Project.toml")
end

# ╔═╡ 3a11733f-8763-45c8-a7ac-af36c53c5118
begin
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 14338fa5-6459-4f15-bff3-71c00104e427
begin
	using DifferentialEquations, Flux, DiffEqFlux, Plots
	
	function lorenz!(du, u, p, t)
	  x, y, z = u
	  α, β, ρ = p
	  du[1] = dx = α*y - α*x
	  du[2] = dy = x*(ρ-z) -y 
	  du[3] = dz = x*y - β*z
	end
end

# ╔═╡ 1e289c0b-a199-4a8f-ad75-dbaf64848091
md"# Inverse Problem."

# ╔═╡ d1ce4471-3cea-4919-8d90-033151c507e7
md" Let us say we have a black box $f(x)$ that models a system. 
>We want to find a value $x$ such that $f(x) = t$."

# ╔═╡ 07f14ebf-f3c0-4e62-b865-01852d0e2bc7
md"It can be formulated as an optimization problem by opimixing the loss 

$$L(x) = |f(x) - t|$$"

# ╔═╡ 01d727b1-2c2e-411c-9785-eee8de705f7d
md" We can use the gradient decent to optimize the loss function if it is differentiable."

# ╔═╡ b0af12ed-45a5-4291-b4f3-be21def53d9c
md" ## Differentiable Programming"

# ╔═╡ 202368d8-aaa5-42ee-bbcc-f2235ad43b07
md" One such example of a black box function is given by a Differntial equation solver. Thanks to the new Automatic differntiation library Zygote.jl, which is now the default AD in flux, it can do source to source automatic differntiation. It can now imlement AD on functions written in Julia, In particular, on any Differntial equation solver. This is in the spirit of **Differntiable programming**, a new paradigm in computation that enable complete code to be differntiated. This is enabling architecture beyond neural networks by implementing different solvers inside as layers."

# ╔═╡ a91b5bc0-d498-42da-8bfc-37758df9e9b7
md" ## Parameter optimization"

# ╔═╡ 695f0dc4-8916-40d1-97fe-827f00b5ce21
md" Let us consider one example of the inverse problem using DiffEqFlux.jl. Consider a system of equations with unknown parameters. 
> Can we find the parameters with low amount of data?

"

# ╔═╡ cc666be8-1f62-4e4b-8c55-34538533efef
md"## Lorenz system"

# ╔═╡ 4a339ac8-eb56-4e1e-ba88-5e6e518267d2
md" Consider the Lorenz equations 
 
$$x' = αy - αx$$
$$y' = x*(ρ-z) -y $$
	  $$ z' = x*y - β*z$$"

# ╔═╡ b281dd8c-aded-464b-84e7-f102145afe00
md" ## Problem"

# ╔═╡ a387d04f-6c17-49b0-b2bb-4a4615936208
md"> What parameters will stabilise $x$ to $5$?"

# ╔═╡ 6bafc1d0-0a36-483a-9266-8290eeaecfea
md" We will start with a random parameter and decent towards the ones that brings $X$ closer to $5$."

# ╔═╡ aca3dae6-568a-4455-9a1a-6893388d7ca9
begin
	u0 = [1.0, 1.0, 1.0]
	tspan = (0.0, 10.0)
	tsteps = 0.0:0.1:10.0
	p = [1.0, 3.0, 2.0]
end

# ╔═╡ 9631fcc1-ed68-4c43-9767-3c7b2757bc42
md"## Visualize"

# ╔═╡ 8bf18077-4439-4215-9627-c7e5dec7fb8e
begin
	prob = ODEProblem(lorenz!, u0, tspan, p)
	sol = solve(prob, Tsit5())
end

# ╔═╡ 2a231c9c-b4a7-4162-9046-f434c82dda63
sol[3,:]

# ╔═╡ aeca5992-a27a-4bad-9f70-1966e14f8b55
begin
	plot(sol)
end

# ╔═╡ 5de08e69-6915-4d6e-ba58-ae1731136122
plot(sol[1,:], sol[2,:], sol[3,:], linewidth = 2, label = "truth", legend = true)

# ╔═╡ cae343d0-c699-4e2f-9add-6ae9c2960b85
begin
	params = Flux.params(p)

function predict_rd() # Our 1-layer "neural network"
  solve(prob,Tsit5(),p=p,saveat=0.1)[1,:] # override with new parameters
end

loss_rd() = sum(abs2,x-5 for x in predict_rd()) # loss function
	
end

# ╔═╡ 6e21412d-88dd-44d2-a223-6160e7e47d6c
begin
	data = Iterators.repeated((), 100)
	opt = ADAM(0.1)
	
	Flux.train!(loss_rd, params, data, opt)
end

# ╔═╡ e02683e3-e8c1-43ad-bc5c-b446f8cef2e7
plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6))

# ╔═╡ 02755dd2-cacb-4353-b398-3162810b47f2
new_sol = solve(prob,Tsit5())

# ╔═╡ e50d5542-649c-4a1b-814d-190062d3456e
p

# ╔═╡ f6ce2f74-a943-4678-a485-c9ea8c6399db
plot(new_sol[1,:], new_sol[2,:], new_sol[3,:], linewidth = 2, label = "truth", legend = true)

# ╔═╡ Cell order:
# ╟─e0eb73a2-9d3a-11eb-3d4f-77229772a63b
# ╟─3a11733f-8763-45c8-a7ac-af36c53c5118
# ╟─1e289c0b-a199-4a8f-ad75-dbaf64848091
# ╟─d1ce4471-3cea-4919-8d90-033151c507e7
# ╟─07f14ebf-f3c0-4e62-b865-01852d0e2bc7
# ╟─01d727b1-2c2e-411c-9785-eee8de705f7d
# ╟─b0af12ed-45a5-4291-b4f3-be21def53d9c
# ╟─202368d8-aaa5-42ee-bbcc-f2235ad43b07
# ╟─a91b5bc0-d498-42da-8bfc-37758df9e9b7
# ╟─695f0dc4-8916-40d1-97fe-827f00b5ce21
# ╟─cc666be8-1f62-4e4b-8c55-34538533efef
# ╟─4a339ac8-eb56-4e1e-ba88-5e6e518267d2
# ╟─b281dd8c-aded-464b-84e7-f102145afe00
# ╟─a387d04f-6c17-49b0-b2bb-4a4615936208
# ╟─6bafc1d0-0a36-483a-9266-8290eeaecfea
# ╠═14338fa5-6459-4f15-bff3-71c00104e427
# ╠═aca3dae6-568a-4455-9a1a-6893388d7ca9
# ╟─9631fcc1-ed68-4c43-9767-3c7b2757bc42
# ╠═8bf18077-4439-4215-9627-c7e5dec7fb8e
# ╠═2a231c9c-b4a7-4162-9046-f434c82dda63
# ╠═aeca5992-a27a-4bad-9f70-1966e14f8b55
# ╠═5de08e69-6915-4d6e-ba58-ae1731136122
# ╠═cae343d0-c699-4e2f-9add-6ae9c2960b85
# ╠═6e21412d-88dd-44d2-a223-6160e7e47d6c
# ╠═e02683e3-e8c1-43ad-bc5c-b446f8cef2e7
# ╠═02755dd2-cacb-4353-b398-3162810b47f2
# ╠═e50d5542-649c-4a1b-814d-190062d3456e
# ╠═f6ce2f74-a943-4678-a485-c9ea8c6399db
