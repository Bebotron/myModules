module QuantumSimulations
	import LinearAlgebra
	export tensor
	export meEvolveState
	export tpEvolveState

	tensor(first, args...) = LinearAlgebra.kron(first, tensor(args...))
	function tensor(arg)
		if ndims(arg[1]) == 0
			output = arg
		else
			output = LinearAlgebra.kron(arg[1], arg[2])
			for i in 3:size(arg,1)
				output = LinearAlgebra.kron(output, arg[i])
			end
		end
		return output
	end

	timePropagator(dt, H, psi) = -im*dt*H*psi

	function tpRK4(dt, H, psi)
		k1 = timePropagator(dt, H, psi)
		k2 = timePropagator(dt, H, psi + 0.5*k1)
		k3 = timePropagator(dt, H, psi + 0.5*k2)
		k4 = timePropagator(dt, H, psi + k3)
		psi += (k1 + 2*k2 + 2*k3 + k4)/6
		return psi
	end

	function lindbladME(H, rho, collapse, coeff)
		newrho = im*(rho*H - H*rho)
		for i in 1:size(coeff,1)
			newrho += coeff[i]*(collapse[i]*rho*collapse[i]' - 0.5*(collapse[i]'collapse[i]*rho + rho*collapse[i]'collapse[i]))
		end
		return newrho
	end

	function meRK4(dt, H, rho, collapse, coeff)
		k1 = lindbladME(H, rho, collapse, coeff)
		k2 = lindbladME(H, rho + k1*dt*0.5, collapse, coeff)
		k3 = lindbladME(H, rho + k2*dt*0.5, collapse, coeff)
		k4 = lindbladME(H, rho + k3*dt, collapse, coeff)
		rho += dt*(k1 + 2*(k2 + k3) + k4)/6
		return rho
	end

	function meEvolveState(initialState, target, H0, collapse, fCOp, Ht, fHam, tvec, timedep)
		dt = tvec[2] - tvec[1]
		dataList = zeros(length(target)*length(initialState) + 1, length(tvec))
		currentState = copy(initialState)
		coeff = zeros(length(collapse))
		Hinit = copy(H0)
		if !timedep
			for i in 1:size(fCOp,1) coeff[i] = fCOp[i] end
			for i in 1:size(fHam,1) Hinit += fHam[i]*Ht[i] end
		end
		for i in 1:length(tvec)
			H = Hinit;
			if timedep
				for j in 1:size(fCOp,1) coeff[j] = fCOp[j, i] end
				for j in 1:size(fHam,1) H += fHam[j, i]*Ht[j] end
			end
			dataList[1, i] = tvec[i]
			for k in 1:length(initialState)
				for j in 1:length(target)
					dataList[(k - 1)*length(target) + j + 1, i] = real(LinearAlgebra.tr(target[j]*currentState[k]'))
				end
				currentState[k] = meRK4(dt, H, currentState[k], collapse, coeff)
			end
		end
		return dataList
	end
	
	function tpEvolveState(initialState, H0, target, Ht, fHam, tvec)
		dt = tvec[2] - tvec[1]
		dataList = zeros(length(target)*length(initialState) + 1, length(tvec))
		currentState = copy(initialState)
		for i in 1:length(tvec)
			H = H0
			for j in 1:size(fHam,1)
				H += fHam[j, i]*Ht[j]
			end
			dataList[1, i] = tvec[i]
			for k in 1:length(initialState)
				for j in 1:length(target)
					dataList[(k - 1)*length(target) + j + 1, i] = abs(target[j]'currentState[k])^2
				end
				currentState[k] = LinearAlgebra.normalize(tpRK4(dt, H, currentState[k]))
			end
		end
		return dataList
	end

end