#pragma once

#include "bound_potential.hpp"
#include <vector>

namespace timemachine {

class Observable {

public:
    virtual void observe(int step, int N, double *d_x_t, double *d_box_t, double lambda) = 0;
};

class AvgPartialUPartialParam : public Observable {

private:
    double *d_du_dp_;
    double *d_s_; // Same size as bp_.size()
    double *d_m_; // Same size as bp_.size()
    BoundPotential *bp_;
    int count_;
    int interval_;

public:
    AvgPartialUPartialParam(BoundPotential *bp, int interval);

    ~AvgPartialUPartialParam();

    virtual void observe(int step, int N, double *d_x_t, double *d_box_t, double lambda) override;

    std::vector<int> shape() const { return this->bp_->shape; }
    // copy into buffer and return shape of params object.
    void avg_du_dp(double *buffer) const;

    void std_du_dp(double *buffer) const;
};

} // namespace timemachine
