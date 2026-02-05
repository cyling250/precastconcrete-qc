import pygmsh
import pyvista as pv

grid = pv.read("output/prefabricated_laminated_beam/mesh-concrete.vtk")
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color="white")
plotter.add_axes(line_width=2)
plotter.show()


grid = pv.read("output/prefabricated_laminated_beam/mesh-rebar.vtk")
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color="white")
plotter.add_axes(line_width=2)
plotter.show()
