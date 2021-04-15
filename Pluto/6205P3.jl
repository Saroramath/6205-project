### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ b251d838-9cc8-11eb-3029-f1e10eb3f697
begin
	using Pkg
	Pkg.status()
	Pkg.activate("Project.toml")
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 3f9b5cf1-780b-440f-92be-36ea90128c57
begin
	using DifferentialEquations, Plots
	g = 10
	acc(dθ,θ,g,t) = -g*sin(θ) + 0.1*θ
	prob = SecondOrderODEProblem(acc,1.0,0.0,(0.0,10.0),g)
	sol = solve(prob)
	plot(sol,label=["Velocity" "Position"])
end

# ╔═╡ 604bd6a5-1bdc-48e7-9a2a-7b7640e7a50d
begin
	using Flux
	NN = Chain(x -> [x],
	           Dense(1,32,tanh),
	           Dense(32,1),
	           first)
		end

# ╔═╡ c5d1197d-d666-410b-9f85-0f27b797e8f6
md"# Physics Informed Neural Networks"

# ╔═╡ f3425c62-8e0b-43da-831d-02d5033cea65
md"## Problem"

# ╔═╡ 52a15a1a-d02b-4527-8b9e-3753889a742f
md" **Let's say we have a swing and we want to model the acceleration at any position**."


# ╔═╡ 63f44cef-48f0-42da-86aa-b290a00dc1ef
md"![Swing](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Tire_swing%2C_near_Litchfield%2C_Connecticut_LCCN2012630791.tif/lossy-page1-800px-Tire_swing%2C_near_Litchfield%2C_Connecticut_LCCN2012630791.tif.jpg)"

# ╔═╡ 975aeecc-38aa-40ef-937f-2ea788a30797
md" We measured the length of the rope, and **took 10 observations** at different angles from the top and noted the acceleration."

# ╔═╡ 64e76291-3198-4d87-8ef0-2eaebbee1d8b
md"## Physics"

# ╔═╡ 7a59c4d4-ef86-494b-87be-dffadfccb53a
md" The classical physics tells us that an Ideal pendulum of length L satisfies the following diffential equation 

$$\frac{d^2 \theta}{dt^2} + \frac{g}{L}\sin \theta = 0$$"

# ╔═╡ 4cfd1ac1-bb07-42da-ad9b-879971f0e4b5
md"![Pendulum](http://www.acs.psu.edu/drussell/Demos/Pendulum/Pendulum.gif)"

# ╔═╡ f04a2c50-d381-4d52-9b59-f3505f9ec528
md"## True Model"

# ╔═╡ b313ca67-7498-4196-8aed-dac384b36978
md" Let us assume the motion of the swing is goverened by a differential equation. Since this is not an ideal pendulum it is likely that it is not following above differential equation but maybe

$$\frac{d^2 \theta}{dt^2} + \frac{g}{L}\sin \theta - 0.1*\theta= 0$$ where $0.1*\theta$ is accounting for other factors. But we of corse are **not aware** of the equation. 

>Can we predict the true acceleration just from those **10 observations**?"

# ╔═╡ 4bff3497-8d28-42e1-b921-33220591ae3d
md" ## Simulating Observations "

# ╔═╡ a9d5fae4-d297-458b-a22f-5ddc36705c75
md" In order to simulate the observations, we will use the true model and solve it suing a numerical ODE solver and observe 10 points."

# ╔═╡ 3a177806-50d3-4c20-a21c-012f3bfe146c
md" Let us assume that $g = 10 m/s^2$ and $L = 1 m$. Hence the equation is 

$$\frac{d^2 \theta}{dt^2} + 10\sin \theta - 0.1*\theta= 0$$ Let us assume that $θ'(0) = 0$ and $θ(0) = 0$"

# ╔═╡ 995afffd-b858-4ca4-bc17-8ea29e58182b
md" Let's use a ODE solver to find the solution."

# ╔═╡ 2fb37c9f-ac03-40d4-b990-7d9fa429fb10
md"### Observations"

# ╔═╡ 04868a73-4892-4a95-8859-deeb3bfb2d2f
begin
	plot_t = 0:0.01:10
	data_plot = sol(plot_t)
	positions_plot = [state[2] for state in data_plot]
	acc_plot = [acc(state[1],state[2],g,plot_t) for state in data_plot]
	
	t = 0:1:10
	dataset = sol(t)
	position_data = [state[2] for state in sol(t)]
	acc_data = [acc(state[1],state[2],g,t) for state in sol(t)]
	
	plot(plot_t,acc_plot,xlabel="t",label="True Acc")
	scatter!(t,acc_data,label="Acc Measurements")
end	

# ╔═╡ e3f61fbe-7863-43b7-85ff-e885e78c0a35
acc_data

# ╔═╡ 2f460533-3323-4434-9440-e484ccae290b
md"## Neural Network"

# ╔═╡ 7c669a76-9c33-4200-ad96-740f07373197
md" Let us train a neural network just on the basis of the observations."

# ╔═╡ 5f31bea3-31e1-41d0-9ebb-165eda700b9d
begin
	loss() = sum(abs2,NN(position_data[i]) - acc_data[i] for i in 1:length(position_data))
	loss()
end

# ╔═╡ 908bda03-b795-4d6e-8025-bc4824c61f75
begin
	display(loss())
	Flux.train!(loss, Flux.params(NN),Iterators.repeated((), 1000), ADAM(0.1))
end

# ╔═╡ 28465d8e-2e6e-45d9-9aa8-e8b5532871b8
begin
	learned_acc_plot = NN.(positions_plot)
	plot(plot_t,acc_plot,xlabel="t",label="True Acc")
	plot!(plot_t,learned_acc_plot,label="Predicted Acc")
	scatter!(t,acc_data,label="Acc Measurements")
end

# ╔═╡ aeda26f2-0fbb-41b9-829a-4cf0f215ccaa
md" We can see that our neural network approximated the supposed acceleration but we are not quite using all he information we have about the system." 

# ╔═╡ c4fb2fe9-e332-4ab6-9a55-9bc0e38feb11
md"### Adding prior physical knowledge"

# ╔═╡ 83d44147-dba8-4b80-bb7c-388682b45715
md" We might not know the actual differential equation governing the swing but we still know that this system must follow the pendulum equation pattern. So we can encode that **knowledge** in the loss function."

# ╔═╡ 4239682b-231a-4bc6-b2d7-e419059c8421
PINN = Chain(x -> [x],
	           Dense(1,32,tanh),
	           Dense(32,1),
	           first)


# ╔═╡ 05e18910-14d1-45dc-ac95-6284870da0df
begin
	random_positions = [2rand()-1 for i in 1:100] # random values in [-1,1]
	loss_ode() = sum(abs2,PINN(x) - (-g*sin(x)) for x in random_positions)
	loss_ode()
end

# ╔═╡ 06128509-05b2-45f2-b951-a1029c3695b2
	lossP() = sum(abs2,NN(position_data[i]) - acc_data[i] for i in 1:length(position_data))

# ╔═╡ 0a027716-c45d-4ba0-a6b6-832fc10683b8
begin
	λ = 0.1
	composed_loss() = lossP() + λ*loss_ode()
end

# ╔═╡ 95505eb8-b9c7-473f-be0c-9628f6f29efa
begin
	Flux.train!(composed_loss, Flux.params(PINN),Iterators.repeated((), 1000), ADAM(0.1))
end

# ╔═╡ 9ec5e1ee-6130-47f0-ab51-960794c01ca5
begin
	Physics_learned_acc_plot = PINN.(positions_plot)
	
	plot(plot_t,acc_plot,xlabel="t",label="True Acc")
	plot!(plot_t,Physics_learned_acc_plot,label="Predicted Acc")
	scatter!(t,acc_data,label="Acc Measurements")
end

# ╔═╡ 8af818b3-ff21-4b8d-ac6e-5247836a66db
md" We can see that this is a quite accurate approximation of the supposed equation."

# ╔═╡ Cell order:
# ╟─b251d838-9cc8-11eb-3029-f1e10eb3f697
# ╟─c5d1197d-d666-410b-9f85-0f27b797e8f6
# ╟─f3425c62-8e0b-43da-831d-02d5033cea65
# ╟─52a15a1a-d02b-4527-8b9e-3753889a742f
# ╟─63f44cef-48f0-42da-86aa-b290a00dc1ef
# ╟─975aeecc-38aa-40ef-937f-2ea788a30797
# ╟─64e76291-3198-4d87-8ef0-2eaebbee1d8b
# ╟─7a59c4d4-ef86-494b-87be-dffadfccb53a
# ╟─4cfd1ac1-bb07-42da-ad9b-879971f0e4b5
# ╟─f04a2c50-d381-4d52-9b59-f3505f9ec528
# ╟─b313ca67-7498-4196-8aed-dac384b36978
# ╟─4bff3497-8d28-42e1-b921-33220591ae3d
# ╟─a9d5fae4-d297-458b-a22f-5ddc36705c75
# ╟─3a177806-50d3-4c20-a21c-012f3bfe146c
# ╟─995afffd-b858-4ca4-bc17-8ea29e58182b
# ╠═3f9b5cf1-780b-440f-92be-36ea90128c57
# ╟─2fb37c9f-ac03-40d4-b990-7d9fa429fb10
# ╠═04868a73-4892-4a95-8859-deeb3bfb2d2f
# ╠═e3f61fbe-7863-43b7-85ff-e885e78c0a35
# ╟─2f460533-3323-4434-9440-e484ccae290b
# ╟─7c669a76-9c33-4200-ad96-740f07373197
# ╠═604bd6a5-1bdc-48e7-9a2a-7b7640e7a50d
# ╠═5f31bea3-31e1-41d0-9ebb-165eda700b9d
# ╠═908bda03-b795-4d6e-8025-bc4824c61f75
# ╠═28465d8e-2e6e-45d9-9aa8-e8b5532871b8
# ╟─aeda26f2-0fbb-41b9-829a-4cf0f215ccaa
# ╟─c4fb2fe9-e332-4ab6-9a55-9bc0e38feb11
# ╟─83d44147-dba8-4b80-bb7c-388682b45715
# ╠═4239682b-231a-4bc6-b2d7-e419059c8421
# ╠═05e18910-14d1-45dc-ac95-6284870da0df
# ╠═06128509-05b2-45f2-b951-a1029c3695b2
# ╠═0a027716-c45d-4ba0-a6b6-832fc10683b8
# ╠═95505eb8-b9c7-473f-be0c-9628f6f29efa
# ╠═9ec5e1ee-6130-47f0-ab51-960794c01ca5
# ╟─8af818b3-ff21-4b8d-ac6e-5247836a66db
