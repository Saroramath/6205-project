### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ d536ad8d-25c2-41cc-baec-847e9bb2cb82
begin
	using Pkg
	Pkg.status()
	Pkg.activate("Project.toml")
end

# ╔═╡ caa2a728-1cbc-4ca8-832f-4507b2feef20
begin
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 0776d7a5-c372-4ca3-8cd0-e8cb3c715f42
begin
	using Flux, DiffEqFlux, DifferentialEquations, Plots
		
		u0 = Float32[2.; 0.]
		datasize = 30
		tspan = (0.0f0,1.5f0)
end

# ╔═╡ b5c47c06-9d2d-11eb-14e3-c52e96c8e8b3
md" # Neural Differential equations"

# ╔═╡ d630679f-e482-4eaf-88c7-eb535a625a42
md"## ResNet"

# ╔═╡ da7f0bf5-e385-4e27-970b-d142304a12e4
md" Let us reconsider the ResNet model. It is a neural network architecture that has direct connections of previous layers with the next layer. Formally 

$$y_{l+1} = y_l + NN(y_l)$$ where $y_l$ is the outpt of the $l^{th}$ layer."

# ╔═╡ 8c4b5fad-e180-4799-ac94-c9353abbb2f2
md"## Euler method"

# ╔═╡ d41f4d45-4344-4184-91d1-0346222f40c4
md" Consider a differential equation 

$$y' = f(y,t.)$$ Euler method is a classical method to numerically solve a differential equation. Let $h$ be a step size, then the iteration is denided as

$$y_n = y_{n-1} + hf(y,t)$$."

# ╔═╡ bf69084d-1844-41ad-a15c-a08efef5ec80
md"## Neural Ordinary Differential Equations"

# ╔═╡ 4c393a7a-976a-4c65-ac0d-e52f3909cbc5
md" The observation that ResNet is an Euler solution with a step-size 1 was noted quite early. This was redecovered in a 2018 in a paper [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) awarded Best Papers of NeuIPS2018. "

# ╔═╡ a81b4a15-f45c-4dbe-b171-f5be728c3725
md" The key premise of the paper is that a discrete Residual neural network can be replaced by an architecture that is parametrised by both $y$ and number of layers $l$.

In addition, since Euler method is prone of large errors, instead of doing Euler approximation as in ResNet, we can utilize some of the **advanced ODE solvers** to train the parameters. Formally it means that a layer is replaced by a ODE solver. The architecture is named as ODENet. "

# ╔═╡ 44e92be8-9fca-4631-a651-c91592b19a87
md"### What about back-propogation?"

# ╔═╡ d623e268-658e-45e5-9396-acd87675dd95
md"The major technical challange in above idea is the following. 
>How to train the ODE-Solver?
Each Neural layer is defines a loss function which can be optimised by gradient decent. Now in place of a layer we have a computer program that need to be 'differentiable' to train. They do this with Adjoint sensitivity analysis. 

"

# ╔═╡ f9e58330-6f90-4a82-bc5e-6e1ef05cb8c1
md"## Equation"

# ╔═╡ e8c5247d-a737-4e49-9ff7-f369e11c9082
md" Thus we are modelling the system as

$$y'(x) = NN(x,t)$$"

# ╔═╡ ea9d697c-e6e7-4efd-88ab-5dd4033ae31f
md"## DiffEqFlux.jl"

# ╔═╡ f0aac252-ddb2-4a46-9d90-85ef88ecdcd9
md" These layers are encoded in the DiffEqFlux.jl as NeuralODE function."

# ╔═╡ 7f81fb2f-8e60-4c2c-a3c7-8a1845d9cf88
md"## Example"

# ╔═╡ 7abeb379-c3e6-4d71-9851-f95acbba7463
md"This code is taken from documentation github repo. We will explain the code here."

# ╔═╡ cef93e63-59de-48f4-a15b-b2aabd7b0667
md" We will consider a set of data points coming from some system goverened by a differntial equations. We will take 30 data points from that system and train an ODE-NET implementation in Julia."

# ╔═╡ df29fcc1-f141-4fd5-8d06-d96f2099f54b
md"### Observations"

# ╔═╡ fe1f2786-ffa2-4887-b22c-561b59124257
md" Observations are noted by using a ODE Solver for the supposed equation."

# ╔═╡ 1c3d3c20-6aa7-446d-ad9d-9662c8d1df76
begin
	function trueODEfunc(du,u,p,t)
		    true_A = [-0.1 2.0; -2.0 -0.1]
		    du .= ((u.^3)'true_A)'
		end
		true_A = [-0.1 2.0; -2.0 -0.1]
		 ((u0.^3)'true_A)'
		t = range(tspan[1],tspan[2],length=datasize)
		prob = ODEProblem(trueODEfunc,u0,tspan)
		ode_data = Array(solve(prob,Tsit5(),saveat=t))
end

# ╔═╡ 73d02c8d-67df-40bf-9222-ea9ae57cd174
md"### Model"

# ╔═╡ b15af78e-a566-4f9c-8f0e-7ea7de033646
md" The model is implemented in order to predict the time series for any given initial position."

# ╔═╡ 79084bc9-d40b-46d6-9868-027c77d7f03b
begin
		dudt = Chain(x -> x.^3,
		             Dense(2,50,tanh),
		             Dense(50,2))
		
		n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)
		ps = Flux.params(n_ode)
		pred = n_ode(u0) # Get the prediction using the correct initial condition
		scatter(t,ode_data[1,:],label="data")
		scatter!(t,pred[1,:],label="prediction")
		
		scatter(t,ode_data[2,:],label="data")
		scatter!(t,pred[2,:],label="prediction")
end

# ╔═╡ 8b07756a-7886-4cf6-9d73-63e589866809
md"### Training the model"

# ╔═╡ c79a9a3b-9c9b-454e-8b1a-e8b8fc0fbfa1
begin
	function predict_n_ode()
		  n_ode(u0)
		end
		loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())
		data = Iterators.repeated((), 1000)
		opt = ADAM(0.1)
		Flux.train!(loss_n_ode, ps, data, opt)
end

# ╔═╡ ee4b32f5-45b2-4ea6-a296-180784f630f4
pred_new = n_ode(u0)

# ╔═╡ c5e668fc-efde-40dd-a118-21a9383be9c1
begin
	pl = scatter(t,ode_data[1,:],label="data")
	  scatter!(pl,t,pred_new[1,:],label="prediction")
	plot(pl)
end

# ╔═╡ Cell order:
# ╟─d536ad8d-25c2-41cc-baec-847e9bb2cb82
# ╟─caa2a728-1cbc-4ca8-832f-4507b2feef20
# ╟─b5c47c06-9d2d-11eb-14e3-c52e96c8e8b3
# ╟─d630679f-e482-4eaf-88c7-eb535a625a42
# ╟─da7f0bf5-e385-4e27-970b-d142304a12e4
# ╟─8c4b5fad-e180-4799-ac94-c9353abbb2f2
# ╟─d41f4d45-4344-4184-91d1-0346222f40c4
# ╟─bf69084d-1844-41ad-a15c-a08efef5ec80
# ╟─4c393a7a-976a-4c65-ac0d-e52f3909cbc5
# ╟─a81b4a15-f45c-4dbe-b171-f5be728c3725
# ╟─44e92be8-9fca-4631-a651-c91592b19a87
# ╟─d623e268-658e-45e5-9396-acd87675dd95
# ╟─f9e58330-6f90-4a82-bc5e-6e1ef05cb8c1
# ╟─e8c5247d-a737-4e49-9ff7-f369e11c9082
# ╟─ea9d697c-e6e7-4efd-88ab-5dd4033ae31f
# ╟─f0aac252-ddb2-4a46-9d90-85ef88ecdcd9
# ╟─7f81fb2f-8e60-4c2c-a3c7-8a1845d9cf88
# ╟─7abeb379-c3e6-4d71-9851-f95acbba7463
# ╟─cef93e63-59de-48f4-a15b-b2aabd7b0667
# ╠═0776d7a5-c372-4ca3-8cd0-e8cb3c715f42
# ╟─df29fcc1-f141-4fd5-8d06-d96f2099f54b
# ╟─fe1f2786-ffa2-4887-b22c-561b59124257
# ╠═1c3d3c20-6aa7-446d-ad9d-9662c8d1df76
# ╟─73d02c8d-67df-40bf-9222-ea9ae57cd174
# ╟─b15af78e-a566-4f9c-8f0e-7ea7de033646
# ╠═79084bc9-d40b-46d6-9868-027c77d7f03b
# ╟─8b07756a-7886-4cf6-9d73-63e589866809
# ╠═c79a9a3b-9c9b-454e-8b1a-e8b8fc0fbfa1
# ╠═ee4b32f5-45b2-4ea6-a296-180784f630f4
# ╠═c5e668fc-efde-40dd-a118-21a9383be9c1
