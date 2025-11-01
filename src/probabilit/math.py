from functools import reduce
import pytensor.tensor as pt

Add = pt.add
Multiply = pt.mul
Max = pt.maximum
Min = pt.minimum
All = lambda *args: reduce(pt.and_, args)
Any = pt.or_
Avg = pt.mean
NoOp = pt.identity
FloorDivide = pt.floor_div
Power = pt.power
Subtract = pt.sub
Equal = pt.eq
NotEqual = pt.neq
LessThan = pt.lt
LessThanOrEqual = pt.le
GreaterThan = pt.gt
GreaterThanOrEqual = pt.ge
IsClose = pt.isclose
IF = pt.where

Negative = pt.neg
Abs = pt.abs
Log = pt.log
Exp = pt.exp
Floor = pt.floor
Ceil = pt.ceil
Sign = pt.sign
Sqrt = pt.sqrt
Square = pt.square
Log10 = pt.log10

Sin = pt.sin
Cos = pt.cos
Tan = pt.tan
Arcsin = pt.arcsin
Arccos = pt.arccos
Arctan = pt.arctan
Arctan2 = pt.arctan2

# Hyperbolic functions
Sinh = pt.sinh
Cosh = pt.cosh
Tanh = pt.tanh
Arcsinh = pt.arcsinh
Arccosh = pt.arccosh
Arctanh = pt.arctanh
Arcsinh = pt.arcsinh
Arccosh = pt.arccosh
Arctanh = pt.arctanh
