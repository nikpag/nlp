#! /usr/bin/env python


a_dim = [
	[0.25, 0.2, 0.3, 0.25],
	[0.2, 0.25, 0.3, 0.25],
	[0.4, 0.2, 0.2, 0.2],
	[0.25, 0.3, 0.2, 0.25]
]

a_mine = [
	[0.15, 0.3, 0.3, 0.25],
	[0.3, 0.15, 0.3, 0.25],
	[0.3, 0.25, 0.15, 0.3],
	[0.25, 0.3, 0.25, 0.2]
]

b_dim = {
	"V": [0.5, 0.8, 0.25, 0.2],
	"U": [0.5, 0.2, 0.75, 0.8]
}

b_mine = {
	"V": [0.6, 0.7, 0.2, 0.25],
	"U": [0.4, 0.3, 0.8, 0.75]
}

p_dim = [0.25, 0.25, 0.25, 0.25]

p_mine = [0.25, 0.25, 0.25, 0.25]

O_dim = list("UVUVVVUUVU")

O_mine = list("UVUVVUVUVU")

d_dim = [
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	]
]

d_mine = [
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	],
	[
		None, None, None, None
	]
]

def func(a, b, p, O, d):
	for t in range(9):
		if t == 0:
			for i in range(4):
				d[0][i] = p[i] * b[O[0]][i]

		for j in range(4):
			d[t+1][j] = max([d[t][i]*a[i][j] for i in range(4)]) * b[O[t+1]][j]
		print(f"d[{t+1}]: {[f'{item:.3}' for item in d[t+1]]}")

# func(a_dim, b_dim, p_dim, O_dim, d_dim)
func(a_mine, b_mine, p_mine, O_mine, d_mine)
