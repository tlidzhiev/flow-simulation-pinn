from typing import Any, Dict, Tuple, Union

from pinns.data.datasets import BoundaryConditionDataset, CollocationDataset, InitialConditionDataset, SimulationDataset


class BaseTrainDataset:
    def __init__(
        self,
        initial_data: Union[InitialConditionDataset, Dict[str, Any]],
        boundary_data: Union[BoundaryConditionDataset, Dict[str, Any]],
        collocation_data: Union[CollocationDataset, Dict[str, Any]],
        simulation_data: Union[SimulationDataset, Dict[str, Any]],
    ):
        self.initial_data = self._to_dataset(initial_data, InitialConditionDataset)
        self.boundary_data = self._to_dataset(boundary_data, BoundaryConditionDataset)
        self.collocation_data = self._to_dataset(collocation_data, CollocationDataset)
        self.simulation_data = self._to_dataset(simulation_data, SimulationDataset)

    def _to_dataset(self, data, dataset_class):
        if not isinstance(data, dataset_class):
            return dataset_class(**data)
        return data

    def to(self, device: str = 'cpu'):
        self.initial_data = self.initial_data.to(device)
        self.boundary_data = self.boundary_data.to(device)
        self.collocation_data = self.collocation_data.to(device)
        self.simulation_data = self.simulation_data.to(device)
        return self

    def get_data(
        self,
    ) -> Tuple[
        InitialConditionDataset,
        BoundaryConditionDataset,
        CollocationDataset,
        SimulationDataset,
    ]:
        return (
            self.initial_data,
            self.boundary_data,
            self.collocation_data,
            self.simulation_data,
        )


class StrongDataset(BaseTrainDataset):
    def __init__(
        self,
        initial_data: Union[InitialConditionDataset, Dict[str, Any]],
        boundary_data: Union[BoundaryConditionDataset, Dict[str, Any]],
        collocation_data: Union[CollocationDataset, Dict[str, Any]],
        simulation_data: Union[SimulationDataset, Dict[str, Any]],
    ):
        super().__init__(
            initial_data=initial_data,
            boundary_data=boundary_data,
            collocation_data=collocation_data,
            simulation_data=simulation_data,
        )


class WeakDataset(BaseTrainDataset):
    def __init__(
        self,
        initial_data: Union[InitialConditionDataset, Dict[str, Any]],
        boundary_data: Union[BoundaryConditionDataset, Dict[str, Any]],
        collocation_data: Union[CollocationDataset, Dict[str, Any]],
        simulation_data: Union[SimulationDataset, Dict[str, Any]],
        quadratures_data,
    ):
        super().__init__(
            initial_data=initial_data,
            boundary_data=boundary_data,
            collocation_data=collocation_data,
            simulation_data=simulation_data,
        )
        self.quadratures_data = quadratures_data

    def to(self, device: str = 'cpu'):
        super().to(device)
        self.quadratures_data = self.quadratures_data.to(device)
        return self

    def get_data(
        self,
    ) -> Tuple[
        InitialConditionDataset,
        BoundaryConditionDataset,
        CollocationDataset,
        SimulationDataset,
    ]:
        return (
            self.initial_data,
            self.boundary_data,
            self.collocation_data,
            self.simulation_data,
            self.quadratures_data,
        )
