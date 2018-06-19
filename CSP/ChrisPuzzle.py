from constraint import *

problem = Problem()

problem.addVariable("red", [1,2,3,4,5])
problem.addVariable("green", [1,2,3,4,5])
problem.addVariable("white", [1,2,3,4,5])
problem.addVariable("yellow", [1,2,3,4,5])
problem.addVariable("blue", [1,2,3,4,5])
problem.addVariable("brit", [1,2,3,4,5])
problem.addVariable("swede", [1,2,3,4,5])
problem.addVariable("dane", [1,2,3,4,5])
problem.addVariable("norwegian", [1,2,3,4,5])
problem.addVariable("german", [1,2,3,4,5])
problem.addVariable("dogs", [1,2,3,4,5])
problem.addVariable("birds", [1,2,3,4,5])
problem.addVariable("cats", [1,2,3,4,5])
problem.addVariable("horses", [1,2,3,4,5])
problem.addVariable("fish", [1,2,3,4,5])
problem.addVariable("tea", [1,2,3,4,5])
problem.addVariable("coffee", [1,2,3,4,5])
problem.addVariable("milk", [1,2,3,4,5])
problem.addVariable("beer", [1,2,3,4,5])
problem.addVariable("water", [1,2,3,4,5])
problem.addVariable("pall", [1,2,3,4,5])
problem.addVariable("dunhill", [1,2,3,4,5])
problem.addVariable("blend", [1,2,3,4,5])
problem.addVariable("bluemaster", [1,2,3,4,5])
problem.addVariable("prince", [1,2,3,4,5])

problem.addConstraint(AllDifferentConstraint(), ("red","green","white","yellow","blue"))
problem.addConstraint(AllDifferentConstraint(), ("brit","swede","dane","norwegian","german"))
problem.addConstraint(AllDifferentConstraint(), ("dogs","birds","cats","horses","fish"))
problem.addConstraint(AllDifferentConstraint(), ("tea","coffee","milk","beer","water"))
problem.addConstraint(AllDifferentConstraint(), ("pall","dunhill","blend","bluemaster","prince"))

problem.addConstraint(lambda a, b: a==b, ("brit", "red"))
problem.addConstraint(lambda a, b: a==b, ("swede", "dogs"))
problem.addConstraint(lambda a, b: a==b, ("dane", "tea"))
problem.addConstraint(lambda a, b: a-1==b, ("green", "white"))
problem.addConstraint(lambda a, b: a==b, ("green", "coffee"))
problem.addConstraint(lambda a, b: a==b, ("pall", "birds"))
problem.addConstraint(lambda a, b: a==b, ("yellow", "dunhill"))
problem.addConstraint(ExactSumConstraint(3), ["milk"])
problem.addConstraint(lambda a, b: a-1==b or a+1==b, ("blend", "cats"))
problem.addConstraint(ExactSumConstraint(1), ["norwegian"])
problem.addConstraint(lambda a, b: a-1==b or a+1==b, ("horses", "dunhill"))
problem.addConstraint(lambda a, b: a==b, ("bluemaster", "beer"))
problem.addConstraint(lambda a, b: a==b, ("german", "prince"))
problem.addConstraint(lambda a, b: a-1==b or a+1==b, ("norwegian", "blue"))
problem.addConstraint(lambda a, b: a-1==b or a+1==b, ("blend", "water"))
print(problem.getSolution())