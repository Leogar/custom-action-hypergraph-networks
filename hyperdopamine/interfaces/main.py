from dof_manipulator import dof_6_manipulator
from manipulatorDrawingTool import ManipulatorDrawingTool

links = [0, 10, 10, 10, 0, 5]
dof_6 = dof_6_manipulator(links)
plotter = ManipulatorDrawingTool(50, links, [1,0,1])
# points = dof_6.generate_points(0, 90, 30)
# plotter.draw(points, save=False, show=True)
plotter.interactive_draw()
