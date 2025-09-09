#include "xue_shan/solver/solver_element.h"

namespace XS::solver {

void element::PyBind(pybind11::module_ &m) {
  pybind11::class_<ElementStretching> element_stretching(m, "ElementStretching");
  element_stretching.def(pybind11::init([](float mu, float lambda, float area, float damping, const Matrix2<float> &Dm,
                                           float sigma_lb, float sigma_ub) {
                           ElementStretching stretching;
                           stretching.mu = mu;
                           stretching.lambda = lambda;
                           stretching.area = area;
                           stretching.damping = damping;
                           stretching.Dm = Dm;
                           stretching.sigma_lb = sigma_lb;
                           stretching.sigma_ub = sigma_ub;
                           return stretching;
                         }),
                         pybind11::arg("mu") = 0.0f, pybind11::arg("lambda") = 0.0f, pybind11::arg("area") = 0.0f,
                         pybind11::arg("damping") = 0.0f, pybind11::arg("Dm") = Matrix2<float>::Identity(),
                         pybind11::arg("sigma_lb") = -1.0f, pybind11::arg("sigma_ub") = -1.0f);
  element_stretching.def("__repr__", [](const ElementStretching &self) {
    Matrix2<float> U, S, Vt;
    SVD(self.Dm, U, S, Vt);
    return pybind11::str(
               "ElementStretching(mu={}, lambda={}, area={}, damping={}, sigma_lb={}, sigma_ub={}, Dm.sigma0={}, "
               "Dm.sigma1={})")
        .format(self.mu, self.lambda, self.area, self.damping, self.sigma_lb, self.sigma_ub, S(0, 0), S(1, 1));
  });
  element_stretching.def_readwrite("mu", &ElementStretching::mu);
  element_stretching.def_readwrite("lambda", &ElementStretching::lambda);
  element_stretching.def_readwrite("area", &ElementStretching::area);
  element_stretching.def_readwrite("damping", &ElementStretching::damping);
  element_stretching.def_readwrite("Dm", &ElementStretching::Dm);
  element_stretching.def_readwrite("sigma_lb", &ElementStretching::sigma_lb);
  element_stretching.def_readwrite("sigma_ub", &ElementStretching::sigma_ub);

  pybind11::class_<ElementBending> element_bending(m, "ElementBending");
  element_bending.def(pybind11::init([](float stiffness, float damping, float theta_rest, float elastic_limit) {
                        ElementBending bending;
                        bending.stiffness = stiffness;
                        bending.damping = damping;
                        bending.theta_rest = theta_rest;
                        bending.elastic_limit = elastic_limit;
                        return bending;
                      }),
                      pybind11::arg("stiffness") = 0.0f, pybind11::arg("damping") = 0.0f,
                      pybind11::arg("theta_rest") = 0.0f, pybind11::arg("elastic_limit") = 4.0f);
  element_bending.def("__repr__", [](const ElementBending &self) {
    return pybind11::str("ElementBending(stiffness={}, damping={}, theta_rest={}, elastic_limit={})")
        .format(self.stiffness, self.damping, self.theta_rest, self.elastic_limit);
  });
  element_bending.def_readwrite("stiffness", &ElementBending::stiffness);
  element_bending.def_readwrite("damping", &ElementBending::damping);
  element_bending.def_readwrite("theta_rest", &ElementBending::theta_rest);
  element_bending.def_readwrite("elastic_limit", &ElementBending::elastic_limit);
}

}  // namespace XS::solver
