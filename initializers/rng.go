package initializers

import "math/rand"

// RNG needs no explanation
type RNG interface {
	Gen() float64
}

type uniform struct {
	lower, upper float64
}

// Uniform returns RNG that gives values uniformly spread between its bounds, which
// can be set by Range
//
//
func Uniform() *uniform {
	return &uniform{defaultValue["uniform-lower"], defaultValue["uniform-upper"]}
}

// Bounds sets the range of a Uniform RNG, returning it.
func (u *uniform) Bounds(lower, upper float64) *uniform {
	u.lower = lower
	u.upper = upper
	return u
}

// Gen is the implementation of RNG for Uniform. It returns a random number.
func (u *uniform) Gen() float64 {
	return rand.Float64()*(u.upper-u.lower) + u.lower
}

type normal struct {
	µ, σ float64
}

// Normal returns an RNG that gives values within a normal distribution. The center
// and standard deviation can be set by Mean and SD, respectively.
//
// Default centers and standard deviations can be set by SetDefault for
// "normal-mean" and "normal-sd".
func Normal() *normal {
	return &normal{defaultValue["normal-mean"], defaultValue["normal-sd"]}
}

// SD sets the value of the standard deviation of the normal distribution.
func (n *normal) SD(sd float64) *normal {
	n.σ = sd
	return n
}

// Mean sets the center of the normal distribution.
func (n *normal) Mean(mean float64) *normal {
	n.µ = mean
	return n
}

// Gen is the implementation of RNG for Normal. It returns a random number.
func (n *normal) Gen() float64 {
	return rand.NormFloat64()*n.σ + n.µ
}

type truncNormal struct {
	*normal
	trunc float64
}

const defaultTrunc float64 = 2.0

// TruncNormal returns an RNG that gives values within an truncated normal
// distribution. The distribution is truncated at 2 standard deviations. The center
// ad standard deviation can be set in the same way as Normal, because Normal is
// embedded in the TruncNormal type.
//
// Additionally, the number of standard deviations to truncate at can be set by
// Trunc.
func TruncNormal() *truncNormal {
	return &truncNormal{Normal(), defaultTrunc}
}

// Trunc sets the number of standard deviations to keep on either side. Trunc will
// panic if given sds <= 0.
func (t *truncNormal) Trunc(sds float64) *truncNormal {
	if sds <= 0 {
		panic("given number of standard deviations to truncate after is <= 0")
	}

	t.trunc = sds
	return t
}

// Gen is the implementation of RNG for TruncNormal. It returns a random number.
func (t *truncNormal) Gen() float64 {
	for {
		v := rand.NormFloat64()
		if v < -t.trunc || v > t.trunc {
			continue
		}

		return v*t.σ + t.µ
	}
}
