#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "BoxesMatch.hpp" // Your C++ header file

namespace py = pybind11;

PYBIND11_MODULE(BoxesMatch_cpp, m) {
    py::class_<BoxObject>(m, "BoxObject")
        .def(py::init<const Box3D&, const std::string&>())
        .def("get_bbox_type", &BoxObject::get_type)
        .def("get_box3d_8_3", &BoxObject::get_box3d_8_3);

    m.def("get_matches_with_score", &getMatchesWithScore, "Calculate matches with score between two sets of 3D boxes.",
          py::arg("infraBoxes"), py::arg("vehicleBoxes"));
}
