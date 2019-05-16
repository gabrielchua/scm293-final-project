# VRPTW

using Gurobi
using JuMP
using CSV

# cd("/Users/gabrielc/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project")

d = CSV.read("test_demand.csv");
d = convert(Array{Int64}, d[:,end]);
tt = CSV.read("test_time.csv");
tt = convert(Array{Int64}, tt[:,2:end]);
C = CSV.read("test_cost.csv");
C = convert(Array{Int64}, C[:,2:end]);
tw = CSV.read("test_timing.csv");
tw = convert(Array{Int64}, tw[:,2:end]);

N = size(d)[1];
n = size(C,2) - 2;
K = 20;
Q = 20;
s = 1;

min_time = tw[:,1];
max_time = tw[:,2];

big_M = 9000;

# for i = 1:n+2
# 	for j = 1:n+2
# 		big_M[i,j] = max(max_time[i] - min_time[j], 0)
# 	end
# end

#big_M = convert(Array{Int64},big_M);

VRP = Model(solver=GurobiSolver(TimeLimit = 36000));
	# Define Variables (X)
	@variable(VRP, X[1:n+2, 1:n+2], Bin);
	@variable(VRP, y[1:n+2]);
	@variable(VRP, W[1:n+2] >= 0);
	#@variable(VRP, K >= 0, Int)

	# Objective Function
	@objective(VRP, Min, sum(C.*X));

	# Constraint - all customers are visited exactly once
	for i = 2:n+1
		@constraint(VRP, sum(X[i,j] for j = 1:n+2) ==  1 + X[i,i])
	end

	# Constraint - correct flow (i.e. if I enter from node i, i must leave from node i)
	for h = 2:n+1
		a = collect(1:1:n+1)
		a = a[a .!= h]
		b = collect(2:1:n+2)
		b = b[b .!= h]
		@constraint(VRP, sum(X[i,h] for i = 1:n+1) - sum(X[h,j] for j = 2:n+2) == 0)
	end

	# Constraint - number of routes/vehicles
	@constraint(VRP, sum(X[i,1] for i = 2:n+1) <= K);
	
	# Constriant - Vehicle Capacity if not exceeded AND no subtours
	for i = 1:n+2
		for j = 1:n+2 
			@constraint(VRP, y[j] >= y[i] + d[j]X[i,j] - Q*(1 - X[i,j]))
		end
	end

	# Constriant - Vehicle Capacity if not exceeded
	@constraint(VRP, d .<= y);
	@constraint(VRP, y .<= Q);


	# Constraints - Time Windows
	for i = 1:n+1
		for j = 2:n+2
			@constraint(VRP, W[j] >= W[i] + ((s + tt[i,j]) * X[i,j]) - (big_M*(1-X[i,j])))
		end
	end

	# # Constraints - Time Windows
	@constraint(VRP, min_time .<= W );
	@constraint(VRP, W .<= max_time);

solve(VRP)

# MathProgBase.numvar(VRP::Model)
# MathProgBase.numconstr(VRP::Model)
# 
# MathProgBase.getsolvetime(m::Model)


# Get Values

X_opt = getvalue(X);
X_opt = convert(Array{Int64},X_opt);

CSV.write("X_opt_test.csv", convert(DataFrame, X_opt))

function first_node(X_opt)
	first_node_list = zeros(0)
	for i = 1:size(C,2)
		if X_opt[1,i] == 1
			append!(first_node_list, i)
		end
	end
	return(convert(Array{Int64},first_node_list))
end

function print_next_node(i)
	for j = 1:size(C,2)
		if X_opt[i, j] == 1
			print(j, " ")
			print_next_node(j)
		end
	end
end

first_nodes = first_node(X_opt);

for i = 1:size(first_nodes)[1]
	print("Route ", i ,": 1 ")
	print(first_nodes[i], " ")
	println(print_next_node(first_nodes[i]))
	println("*******")
end


function return_next_node(i)
	for j = 1:size(C,2)
		if X_opt[i, j] == 1
			return(j)
		end
	end
end

function loop_next_node(i)
	 = zeros(0)
	temp = return_next_node(i)
	append!(S, temp)
	loop_next_node(temp)
	return(S)
end




Routes = Dict()
Routes[1] = [1 7 144 153 6 173 179 221]
Routes[2] = [1 11 30 130 89 83 72 128 70 112 162 118 68 64 217 205 150 117 196 194 221]
Routes[3] = [1 19 63 97 2 115 82 156 207 221]
Routes[4] = [1 49 71 122 152 23 77 25 120 176 9 145 87 93 197 216 133 214 129 199 221]
Routes[5] = [1 60 38 110 116 79 212 113 108 28 127 106 132 78 154 65 73 221]
Routes[6] = [1 90 36 215 201 62 175 86 41 45 12 43 119 139 33 22 195 47 39 16 151 221]
Routes[7] = [1 114 42 186 5 181 91 37 13 169 147 221]
Routes[8] = [1 123 157 210 206 193 221]
Routes[9] = [1 126 124 10 182 143 178 21 4 48 75 155 190 185 189 14 183 221]
Routes[10] = [1 135 188 104 140 204 208 51 58 66 180 158 159 149 146 138 136 172 221]
Routes[11] = [1 161 20 50 142 111 101 164 209 95 203 85 15 160 218 168 8 202 221]
Routes[12] = [1 166 125 105 35 121 191 163 221]
Routes[13] = [1 170 192 34 46 219 3 80 32 107 167 69 67 44 134 17 131 198 109 165 187 221]
Routes[14] = [1 174 74 56 103 40 99 102 184 61 84 200 57 96 100 213 177 55 59 52 171 221]
Routes[15] = [1 220 137 31 211 148 81 18 53 92 76 54 27 24 98 26 88 29 94 141 221]


Routes_json = JSON.json(Routes)

# write the file with the stringdata variable information
open("11_test.json", "w") do f
        write(f, Routes_json)
     end

