### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 91f246b9-cba8-4726-bd1c-d5c1edbfeeda
using Pkg

# ╔═╡ f9481c79-b977-40b0-92d0-b415ae29d47a
Pkg.activate("Project.toml")

# ╔═╡ a7ec611d-4d0f-47f5-81cf-fe8e4ae4d0de
using PlutoUI

# ╔═╡ 8e773264-3d77-44ca-b6ed-021dffa39f1b
using Flux

# ╔═╡ 2fd70d8c-2a73-491c-a655-218efca32595
# to check the status of pkgs shown in REPL 
Pkg.status()

# ╔═╡ 8a612d80-9dc7-40e7-8631-e0699eecacd7
#Pkg.add("PlutoUI")

# ╔═╡ becb29c8-adb1-4476-a4b3-7899b020bcc9


# ╔═╡ 41b2c3d5-6941-45cd-9422-ebb240a2fdb4
PlutoUI.TableOfContents()

# ╔═╡ cda88fdc-21c5-4849-a35d-8c812c7f4bc4
md"# Julia"

# ╔═╡ fe258735-790f-45f4-b182-3e0d9b6a3673
a2 = [1;4;5]

# ╔═╡ 47e17aa5-12b8-4445-9223-7ec77699529b
typeof(a2)

# ╔═╡ 6aa9407a-8e06-4c7f-9218-0f84ed50e69c
md"## Operations"

# ╔═╡ 5c942d8e-adfc-4163-9d02-eba4b58abc2d
sum(1:3)

# ╔═╡ e92b9da6-8fcb-4648-81c6-346a967b5bdc
sum(abs2,1:3)

# ╔═╡ 087b7d4f-1648-4f11-8a8b-6cffca4fe33b
md" ## Arrays"

# ╔═╡ 02642d60-7eb4-497d-a0b1-8b539d254345
for i in 1:7
	println(i)
end

# ╔═╡ 46f240ed-b10c-43f0-b2b7-c575200c592a
a1 = 1:3

# ╔═╡ d987060a-31fa-49fd-8106-03a3931f3962
typeof(a1)

# ╔═╡ 5621ea5b-2ee0-4dde-bdd1-7be36c880a09
A1 = rand(2,2)

# ╔═╡ b1852727-5039-4e97-bb39-bb13067cb13b
sum(abs2,A1)

# ╔═╡ 25694c3e-0c58-4e6c-bcd3-da939bebac92
A = [1 2; 3 4]

# ╔═╡ fe9240e5-7af4-40c8-9b84-6ffc5f7a928f
sum(abs2,A)

# ╔═╡ 407a6bba-2396-448b-a411-7f357df39fe9
A * A

# ╔═╡ ac3e5696-aa8e-4cb9-88c3-40058a7a01ec
A .* A

# ╔═╡ 104bb99d-3de3-460d-acce-b74f175261eb
A^2

# ╔═╡ f012f657-50b8-4c38-b0f9-6ebae9b2b0d0
md"# Machine learning"

# ╔═╡ 05083238-ad9a-4397-b600-57349151c3bd
Pkg.status()

# ╔═╡ 18267317-5144-4790-b19b-c76f8eebb4d4
md"## Neural Network"

# ╔═╡ 401201de-f493-4da8-9eb2-0b03c62170d8
NN = Chain(Dense(10,32,tanh), Dense(32,32,tanh),Dense(32,5))

# ╔═╡ cff7a55d-58ea-4827-aa99-abc480eaab29
begin
	x = rand(10)
	
	NN(x)
end

# ╔═╡ 53a1ef69-912a-4595-84f5-d6fdcf06708c
NN[1].W

# ╔═╡ 1aaea20e-fd3f-4e41-8e71-ce91e063474a
typeof(NN[1].W)

# ╔═╡ 86d5d579-fe79-4b17-b6e4-2d39986170e6
sum(abs2,NN(rand(10)).-1)

# ╔═╡ 0e53d782-afe2-4912-876c-73e95aad2225


# ╔═╡ b04dcc9b-255a-40c3-8330-0eba0e8b83b5
p = params(NN)

# ╔═╡ cf541c7e-d522-467e-8a08-df13d6eac104


# ╔═╡ 7cfb0fa7-ca9c-4d21-b856-35b592bb51dc
md"# Differential Equation and Neural Network"

# ╔═╡ 635f379b-a364-42d4-a9c7-8fa33a258b2b
begin
	NNODE = Chain(x -> [x], # Take in a scalar and transform it into an array
	           Dense(1,32,tanh),
	           Dense(32,1),
	           first) # Take first value, i.e. return a scalar
	NNODE(1.0)
end

# ╔═╡ 0328ad85-b401-447e-b36a-a24ae5b5060c
begin
	g(t) = t*NNODE(t) + 1f0
	using Statistics
	ϵ = sqrt(eps(Float32))
	loss() = mean(abs2(((g(t+ϵ)-g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)
	
end

# ╔═╡ 83f4bb2a-e4d9-4bd2-aadf-5e96f2f2209d
Flux.train!(loss,p,Iterators.repeated((),1000), ADAM(0.1))

# ╔═╡ cb98342e-f37d-47fe-bd2e-d5641e73dbf4
begin
	using Plots
	t = 0:0.001:1.0
	plot(t,g.(t),label="NN")
	plot!(t,1.0 .+ sin.(2π.*t)/2π, label = "True Solution")
end

# ╔═╡ f7931515-0f0b-4c40-8b6b-5d62ce93c3a3
begin
	opt = Flux.Descent(0.01)
	data = Iterators.repeated((), 5000)
	iter = 0
	cb = function () #callback function to observe training
	  global iter += 1
	  if iter % 500 == 0
	    display(loss())
	  end
	end
	display(loss())
	Flux.train!(loss, Flux.params(NNODE), data, opt; cb=cb)
end

# ╔═╡ 265998b3-588c-454d-beee-7c3f4920bf67
dg(t)  = gradient(g, t)[1]

# ╔═╡ 5c552239-5dcc-4938-a828-9ce94ba7ca97
dg(1)

# ╔═╡ 1746d73e-5352-4540-b194-0705252efe80
loss2() = mean(abs2(dg(t) - cos(2π*t)) for t in 0:1f-2:1f0)

# ╔═╡ d792b0bd-c58b-4f04-a130-eaaa5ca57148
loss() 

# ╔═╡ 3cc16bb5-cb74-4ae4-bccb-bf2be428e074
loss2()

# ╔═╡ 2f1b3730-9a59-4878-adb7-7163c3c14860
md"# Hook's law"

# ╔═╡ 51c25e5a-debd-4868-b8e4-2ab0a5fe1a9d


# ╔═╡ Cell order:
# ╠═2fd70d8c-2a73-491c-a655-218efca32595
# ╠═8a612d80-9dc7-40e7-8631-e0699eecacd7
# ╠═becb29c8-adb1-4476-a4b3-7899b020bcc9
# ╠═a7ec611d-4d0f-47f5-81cf-fe8e4ae4d0de
# ╠═41b2c3d5-6941-45cd-9422-ebb240a2fdb4
# ╟─cda88fdc-21c5-4849-a35d-8c812c7f4bc4
# ╠═fe258735-790f-45f4-b182-3e0d9b6a3673
# ╠═47e17aa5-12b8-4445-9223-7ec77699529b
# ╟─6aa9407a-8e06-4c7f-9218-0f84ed50e69c
# ╠═5c942d8e-adfc-4163-9d02-eba4b58abc2d
# ╠═e92b9da6-8fcb-4648-81c6-346a967b5bdc
# ╟─087b7d4f-1648-4f11-8a8b-6cffca4fe33b
# ╠═02642d60-7eb4-497d-a0b1-8b539d254345
# ╠═46f240ed-b10c-43f0-b2b7-c575200c592a
# ╠═d987060a-31fa-49fd-8106-03a3931f3962
# ╠═5621ea5b-2ee0-4dde-bdd1-7be36c880a09
# ╠═b1852727-5039-4e97-bb39-bb13067cb13b
# ╠═25694c3e-0c58-4e6c-bcd3-da939bebac92
# ╠═fe9240e5-7af4-40c8-9b84-6ffc5f7a928f
# ╠═407a6bba-2396-448b-a411-7f357df39fe9
# ╠═ac3e5696-aa8e-4cb9-88c3-40058a7a01ec
# ╠═104bb99d-3de3-460d-acce-b74f175261eb
# ╟─f012f657-50b8-4c38-b0f9-6ebae9b2b0d0
# ╠═91f246b9-cba8-4726-bd1c-d5c1edbfeeda
# ╠═f9481c79-b977-40b0-92d0-b415ae29d47a
# ╠═05083238-ad9a-4397-b600-57349151c3bd
# ╠═8e773264-3d77-44ca-b6ed-021dffa39f1b
# ╟─18267317-5144-4790-b19b-c76f8eebb4d4
# ╠═401201de-f493-4da8-9eb2-0b03c62170d8
# ╠═cff7a55d-58ea-4827-aa99-abc480eaab29
# ╠═53a1ef69-912a-4595-84f5-d6fdcf06708c
# ╠═1aaea20e-fd3f-4e41-8e71-ce91e063474a
# ╠═86d5d579-fe79-4b17-b6e4-2d39986170e6
# ╠═0e53d782-afe2-4912-876c-73e95aad2225
# ╠═b04dcc9b-255a-40c3-8330-0eba0e8b83b5
# ╠═83f4bb2a-e4d9-4bd2-aadf-5e96f2f2209d
# ╠═cf541c7e-d522-467e-8a08-df13d6eac104
# ╟─7cfb0fa7-ca9c-4d21-b856-35b592bb51dc
# ╠═635f379b-a364-42d4-a9c7-8fa33a258b2b
# ╠═f7931515-0f0b-4c40-8b6b-5d62ce93c3a3
# ╠═0328ad85-b401-447e-b36a-a24ae5b5060c
# ╠═cb98342e-f37d-47fe-bd2e-d5641e73dbf4
# ╠═265998b3-588c-454d-beee-7c3f4920bf67
# ╠═5c552239-5dcc-4938-a828-9ce94ba7ca97
# ╠═1746d73e-5352-4540-b194-0705252efe80
# ╠═d792b0bd-c58b-4f04-a130-eaaa5ca57148
# ╠═3cc16bb5-cb74-4ae4-bccb-bf2be428e074
# ╟─2f1b3730-9a59-4878-adb7-7163c3c14860
# ╠═51c25e5a-debd-4868-b8e4-2ab0a5fe1a9d
