## R1CS
这是一个约束系统，If we have an arithmetic circuit with signals s_1,...,s_n, then we define a constraint as an equation of the following form:
`(a_1*s_1 + ... + a_n*s_n) * (b_1*s_1 + ... + b_n*s_n) + (c_1*s_1 + ... + c_n*s_n) = 0`
>circom allows programmers to define the constraints that define the arithmetic circuit. All constraints must be quadratic of the form A*B + C = 0,
>where A, B and C are linear combinations of signals. circom will apply some minor transformations on the defined constraints in order to meet the format A*B + C = 0:

如何生成约束
- HDL：circom
- Library: arkworks(rust)
- PL: ZoKrates