"""Boto3Cloud against botocore's Stubber: request shapes and error handling, no network."""

from collections.abc import Iterator
from typing import Any

import pytest
from botocore.stub import ANY, Stubber

from tests.unit.cloud.conftest import ACCOUNT, REGION
from vms.cloud.aws import (
    Boto3Cloud,
    BudgetSpec,
    Image,
    LaunchDeniedError,
    LaunchSpec,
    resource_of,
)
from vms.cloud.settings import Timeouts

SPEC = LaunchSpec(
    image=Image(id="ami-1", name="DLAMI", root_device="/dev/sda1"),
    instance_type="g6.xlarge",
    key_name="vms-gpu-x",
    security_group_id="sg-1",
    root_volume_gb=100,
    user_data="#!/bin/bash\nshutdown -h +150\n",
    tags={"project": "vms", "session": "x"},
)


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> Boto3Cloud:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    return Boto3Cloud(REGION, Timeouts())


@pytest.fixture
def ec2(cloud: Boto3Cloud) -> Iterator[Stubber]:
    with Stubber(cloud._ec2) as stub:
        yield stub
        stub.assert_no_pending_responses()


def client(cloud: Boto3Cloud, service: str, region: str | None = None) -> Any:
    return cloud._client(service, region)


def test_resource_of_parses_ec2_arns() -> None:
    assert resource_of(f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/i-1") == ("instance", "i-1")
    assert resource_of(f"arn:aws:ec2:{REGION}:{ACCOUNT}:key-pair/key-1") == ("key-pair", "key-1")


def test_the_launch_is_on_demand_tagged_encrypted_and_terminates_on_shutdown(
    cloud: Boto3Cloud, ec2: Stubber
) -> None:
    tags = [{"Key": "project", "Value": "vms"}, {"Key": "session", "Value": "x"}]
    ec2.add_response(
        "run_instances",
        {"Instances": [{"InstanceId": "i-1"}]},
        {
            "ImageId": "ami-1",
            "InstanceType": "g6.xlarge",
            "MinCount": 1,
            "MaxCount": 1,
            "KeyName": "vms-gpu-x",
            "SecurityGroupIds": ["sg-1"],
            "UserData": SPEC.user_data,
            "InstanceInitiatedShutdownBehavior": "terminate",
            "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": 100,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                        "Encrypted": True,
                    },
                }
            ],
            "TagSpecifications": [
                {"ResourceType": "instance", "Tags": tags},
                {"ResourceType": "volume", "Tags": tags},
                {"ResourceType": "network-interface", "Tags": tags},
            ],
        },
    )
    assert cloud.run_instance(SPEC) == "i-1"


def test_dry_run_success_is_the_dry_run_error(cloud: Boto3Cloud, ec2: Stubber) -> None:
    ec2.add_client_error("run_instances", service_error_code="DryRunOperation")
    cloud.check_launch(SPEC)


def test_dry_run_denial_raises(cloud: Boto3Cloud, ec2: Stubber) -> None:
    ec2.add_client_error(
        "run_instances",
        service_error_code="UnauthorizedOperation",
        service_message="explicit deny in a service control policy",
    )
    with pytest.raises(LaunchDeniedError, match="service control policy"):
        cloud.check_launch(SPEC)


def test_terminated_or_missing_instances_do_not_exist(cloud: Boto3Cloud, ec2: Stubber) -> None:
    def reply(state: str) -> dict[str, Any]:
        item = {"InstanceId": "i-1", "State": {"Name": state, "Code": 0}}
        return {"Reservations": [{"Instances": [item]}]}

    ec2.add_response("describe_instances", reply("terminated"), {"InstanceIds": ["i-1"]})
    ec2.add_response("describe_instances", reply("shutting-down"), {"InstanceIds": ["i-1"]})
    ec2.add_client_error("describe_instances", service_error_code="InvalidInstanceID.NotFound")
    instance = f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/i-1"
    assert cloud.exists(instance) is False
    assert cloud.exists(instance) is True  # still shutting down: not gone yet
    assert cloud.exists(instance) is False


def test_exists_for_other_resource_types(cloud: Boto3Cloud, ec2: Stubber) -> None:
    ec2.add_client_error("describe_security_groups", service_error_code="InvalidGroup.NotFound")
    ec2.add_response("describe_key_pairs", {"KeyPairs": [{"KeyPairId": "key-1"}]})
    ec2.add_response("describe_volumes", {"Volumes": []})
    prefix = f"arn:aws:ec2:{REGION}:{ACCOUNT}"
    assert cloud.exists(f"{prefix}:security-group/sg-1") is False
    assert cloud.exists(f"{prefix}:key-pair/key-1") is True
    assert cloud.exists(f"{prefix}:volume/vol-1") is False
    # A type the check does not know is assumed to exist (fail loudly, not silently).
    assert cloud.exists(f"{prefix}:elastic-ip/eipalloc-1") is True


def test_security_group_in_use_is_retried_not_raised(cloud: Boto3Cloud, ec2: Stubber) -> None:
    ec2.add_client_error("delete_security_group", service_error_code="DependencyViolation")
    ec2.add_client_error("delete_security_group", service_error_code="InvalidGroup.NotFound")
    ec2.add_client_error("delete_security_group", service_error_code="UnauthorizedOperation")
    assert cloud.delete_security_group("sg-1") is False
    assert cloud.delete_security_group("sg-1") is True
    with pytest.raises(Exception, match="UnauthorizedOperation"):
        cloud.delete_security_group("sg-1")


def test_tagged_arns_filters_on_every_tag(cloud: Boto3Cloud) -> None:
    tagging = client(cloud, "resourcegroupstaggingapi")
    with Stubber(tagging) as stub:
        stub.add_response(
            "get_resources",
            {"ResourceTagMappingList": [{"ResourceARN": "arn:1"}, {"ResourceARN": "arn:2"}]},
            {
                "TagFilters": [
                    {"Key": "project", "Values": ["vms"]},
                    {"Key": "session", "Values": ["x"]},
                ]
            },
        )
        assert cloud.tagged_arns({"session": "x", "project": "vms"}) == ["arn:1", "arn:2"]


def test_budget_has_actual_and_forecast_alerts(cloud: Boto3Cloud) -> None:
    budgets = client(cloud, "budgets", "us-east-1")
    spec = BudgetSpec(
        name="vms-monthly",
        amount_usd=25,
        email="ops@example.com",
        actual_thresholds_pct=[50, 100],
        forecast_thresholds_pct=[100],
        tags={"project": "vms"},
    )
    with Stubber(budgets) as stub:
        stub.add_client_error("describe_budget", service_error_code="NotFoundException")
        stub.add_response(
            "create_budget",
            {},
            {
                "AccountId": ACCOUNT,
                "Budget": {
                    "BudgetName": "vms-monthly",
                    "BudgetLimit": {"Amount": "25.00", "Unit": "USD"},
                    "TimeUnit": "MONTHLY",
                    "BudgetType": "COST",
                    "CostTypes": {"IncludeCredit": False, "IncludeRefund": False},
                },
                "NotificationsWithSubscribers": ANY,
                "ResourceTags": [{"Key": "project", "Value": "vms"}],
            },
        )
        assert cloud.budget_exists(ACCOUNT, "vms-monthly") is False
        cloud.create_budget(ACCOUNT, spec)


def test_budget_notifications() -> None:
    # The notification list itself, checked without the stub's ANY.
    from vms.cloud import aws

    captured: dict[str, Any] = {}

    class Budgets:
        def create_budget(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    cloud = object.__new__(aws.Boto3Cloud)
    cloud._budgets = lambda: Budgets()  # type: ignore[method-assign]
    spec = BudgetSpec("b", 25, "ops@example.com", [50, 100], [100])
    cloud.create_budget(ACCOUNT, spec)
    kinds = [
        (n["Notification"]["NotificationType"], n["Notification"]["Threshold"])
        for n in captured["NotificationsWithSubscribers"]
    ]
    assert kinds == [("ACTUAL", 50), ("ACTUAL", 100), ("FORECASTED", 100)]
    subscribers = captured["NotificationsWithSubscribers"][0]["Subscribers"]
    assert subscribers == [{"SubscriptionType": "EMAIL", "Address": "ops@example.com"}]
