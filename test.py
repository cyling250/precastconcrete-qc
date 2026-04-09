import pygmsh
import pyvista as pv

grid_concrete = pv.read("output/prefabricated_laminated_beam/mesh-concrete.vtk")
grid_rebar = pv.read("output/prefabricated_laminated_beam/mesh-rebar.vtk")
plotter = pv.Plotter()
plotter.add_mesh(grid_concrete, show_edges=True, color="red", opacity=0.9)
plotter.add_mesh(grid_rebar, show_edges=True, color="lightgray", opacity=0.7)
plotter.add_axes(line_width=2)
plotter.show()
