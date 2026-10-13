"""The AWS calls a GPU session makes, behind a small interface (`Cloud`) so tests use a fake.

`Boto3Cloud` is the real implementation: EC2 in the project's Region, Service Quotas, SSM (the
AMI parameter), the Resource Groups Tagging API (the leftover check) and AWS Budgets. Every
client has connect and read timeouts and botocore's standard retry mode (bounded attempts,
exponential backoff with jitter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from vms.cloud.settings import Timeouts


@dataclass(frozen=True)
class Identity:
    account: str
    arn: str
    region: str


@dataclass(frozen=True)
class Image:
    id: str
    name: str
    root_device: str


@dataclass(frozen=True)
class LaunchSpec:
    image: Image
    instance_type: str
    key_name: str
    security_group_id: str
    root_volume_gb: int
    user_data: str
    tags: dict[str, str]


@dataclass(frozen=True)
class Instance:
    id: str
    state: str
    public_ip: str | None = None
    launch_time: datetime | None = None


@dataclass(frozen=True)
class BudgetSpec:
    name: str
    amount_usd: float
    email: str
    actual_thresholds_pct: list[float] = field(default_factory=list)
    forecast_thresholds_pct: list[float] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)


class LaunchDeniedError(Exception):
    """A dry-run launch failed: permissions, a service control policy or a parameter."""


class CloudError(Exception):
    """An AWS call failed in a way the session cannot recover from."""


class Cloud(Protocol):
    def identity(self) -> Identity: ...
    def vcpu_quota(self, quota_code: str) -> float: ...
    def instance_vcpus(self, instance_type: str) -> int: ...
    def resolve_image(self, parameter: str) -> Image: ...
    def check_launch(self, spec: LaunchSpec) -> None:
        """Dry-run the launch; raises `LaunchDeniedError` when it would fail."""
        ...

    def import_key_pair(self, name: str, public_key: bytes, tags: dict[str, str]) -> str: ...
    def create_security_group(self, name: str, ssh_cidr: str, tags: dict[str, str]) -> str: ...
    def run_instance(self, spec: LaunchSpec) -> str: ...
    def describe_instance(self, instance_id: str) -> Instance | None:
        """None when the instance does not exist (any more)."""
        ...

    def terminate_instance(self, instance_id: str) -> None: ...
    def delete_security_group(self, group_id: str) -> bool:
        """True when deleted or already gone; False while still in use (retry later)."""
        ...

    def delete_key_pair(self, key_pair_id: str) -> None: ...
    def tagged_arns(self, tags: dict[str, str]) -> list[str]: ...
    def exists(self, arn: str) -> bool:
        """Whether the resource behind an ARN still exists (terminated instances do not)."""
        ...

    def budget_exists(self, account: str, name: str) -> bool: ...
    def create_budget(self, account: str, spec: BudgetSpec) -> None: ...


def resource_of(arn: str) -> tuple[str, str]:
    """`arn:aws:ec2:<region>:<account>:instance/i-1` -> ("instance", "i-1")."""
    resource = arn.split(":", 5)[-1]
    kind, _, ident = resource.partition("/")
    return kind, ident


def _code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _not_found(exc: ClientError) -> bool:
    return _code(exc).endswith("NotFound") or _code(exc) == "NotFoundException"


def _tag_list(tags: dict[str, str]) -> list[dict[str, str]]:
    return [{"Key": k, "Value": v} for k, v in sorted(tags.items())]


class Boto3Cloud:
    """`Cloud` on boto3. `region` None uses the AWS configuration's Region."""

    def __init__(self, region: str | None, timeouts: Timeouts, profile: str | None = None):
        self._session = boto3.Session(profile_name=profile or None, region_name=region or None)
        if not self._session.region_name:
            raise CloudError("no AWS Region configured: set AWS_REGION or the profile's region")
        self.region: str = self._session.region_name
        self._config = Config(
            connect_timeout=timeouts.aws_connect_s,
            read_timeout=timeouts.aws_read_s,
            retries={"mode": "standard", "max_attempts": timeouts.aws_max_attempts},
        )
        self._clients: dict[tuple[str, str], Any] = {}
        self._ec2: Any = self._client("ec2")

    def _client(self, service: str, region: str | None = None) -> Any:
        key = (service, region or self.region)
        if key not in self._clients:
            self._clients[key] = self._session.client(  # type: ignore[call-overload]
                service, region_name=key[1], config=self._config
            )
        return self._clients[key]

    def identity(self) -> Identity:
        me = self._client("sts").get_caller_identity()
        return Identity(account=me["Account"], arn=me["Arn"], region=self.region)

    def vcpu_quota(self, quota_code: str) -> float:
        quota = self._client("service-quotas").get_service_quota(
            ServiceCode="ec2", QuotaCode=quota_code
        )
        return float(quota["Quota"]["Value"])

    def instance_vcpus(self, instance_type: str) -> int:
        types = self._ec2.describe_instance_types(InstanceTypes=[instance_type])
        return int(types["InstanceTypes"][0]["VCpuInfo"]["DefaultVCpus"])

    def resolve_image(self, parameter: str) -> Image:
        image_id = self._client("ssm").get_parameter(Name=parameter)["Parameter"]["Value"]
        image = self._ec2.describe_images(ImageIds=[image_id])["Images"][0]
        return Image(id=image_id, name=image["Name"], root_device=image["RootDeviceName"])

    def _run_params(self, spec: LaunchSpec) -> dict[str, Any]:
        tags = _tag_list(spec.tags)
        return {
            "ImageId": spec.image.id,
            "InstanceType": spec.instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "KeyName": spec.key_name,
            "SecurityGroupIds": [spec.security_group_id],
            "UserData": spec.user_data,
            # A shutdown from inside (the max_session_minutes timer) terminates, not stops.
            "InstanceInitiatedShutdownBehavior": "terminate",
            "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
            "BlockDeviceMappings": [
                {
                    "DeviceName": spec.image.root_device,
                    "Ebs": {
                        "VolumeSize": spec.root_volume_gb,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                        "Encrypted": True,
                    },
                }
            ],
            "TagSpecifications": [
                {"ResourceType": kind, "Tags": tags}
                for kind in ("instance", "volume", "network-interface")
            ],
        }

    def check_launch(self, spec: LaunchSpec) -> None:
        params = self._run_params(spec)
        # The key pair and security group do not exist yet at preflight.
        for key in ("KeyName", "SecurityGroupIds"):
            params.pop(key)
        try:
            self._ec2.run_instances(DryRun=True, **params)
        except ClientError as exc:
            if _code(exc) == "DryRunOperation":
                return
            raise LaunchDeniedError(str(exc)) from exc
        raise LaunchDeniedError("dry run unexpectedly launched nothing and reported no error")

    def import_key_pair(self, name: str, public_key: bytes, tags: dict[str, str]) -> str:
        made = self._ec2.import_key_pair(
            KeyName=name,
            PublicKeyMaterial=public_key,
            TagSpecifications=[{"ResourceType": "key-pair", "Tags": _tag_list(tags)}],
        )
        return str(made["KeyPairId"])

    def create_security_group(self, name: str, ssh_cidr: str, tags: dict[str, str]) -> str:
        made = self._ec2.create_security_group(
            GroupName=name,
            Description="VMS GPU session: SSH from the operator only",
            TagSpecifications=[{"ResourceType": "security-group", "Tags": _tag_list(tags)}],
        )
        group_id = str(made["GroupId"])
        self._ec2.authorize_security_group_ingress(
            GroupId=group_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": ssh_cidr, "Description": "operator"}],
                }
            ],
        )
        return group_id

    def run_instance(self, spec: LaunchSpec) -> str:
        launched = self._ec2.run_instances(**self._run_params(spec))
        return str(launched["Instances"][0]["InstanceId"])

    def describe_instance(self, instance_id: str) -> Instance | None:
        try:
            found = self._ec2.describe_instances(InstanceIds=[instance_id])
        except ClientError as exc:
            if _not_found(exc):
                return None
            raise
        for reservation in found["Reservations"]:
            for item in reservation["Instances"]:
                return Instance(
                    id=item["InstanceId"],
                    state=item["State"]["Name"],
                    public_ip=item.get("PublicIpAddress"),
                    launch_time=item.get("LaunchTime"),
                )
        return None

    def terminate_instance(self, instance_id: str) -> None:
        try:
            self._ec2.terminate_instances(InstanceIds=[instance_id])
        except ClientError as exc:
            if not _not_found(exc):
                raise

    def delete_security_group(self, group_id: str) -> bool:
        try:
            self._ec2.delete_security_group(GroupId=group_id)
        except ClientError as exc:
            if _not_found(exc):
                return True
            if _code(exc) == "DependencyViolation":
                return False
            raise
        return True

    def delete_key_pair(self, key_pair_id: str) -> None:
        try:
            self._ec2.delete_key_pair(KeyPairId=key_pair_id)
        except ClientError as exc:
            if not _not_found(exc):
                raise

    def tagged_arns(self, tags: dict[str, str]) -> list[str]:
        pages = self._client("resourcegroupstaggingapi").get_paginator("get_resources")
        filters = [{"Key": k, "Values": [v]} for k, v in sorted(tags.items())]
        return [
            item["ResourceARN"]
            for page in pages.paginate(TagFilters=filters)
            for item in page["ResourceTagMappingList"]
        ]

    def exists(self, arn: str) -> bool:
        kind, ident = resource_of(arn)
        if kind == "instance":
            instance = self.describe_instance(ident)
            return instance is not None and instance.state != "terminated"
        describe: dict[str, tuple[str, str, str]] = {
            "volume": ("describe_volumes", "VolumeIds", "Volumes"),
            "security-group": ("describe_security_groups", "GroupIds", "SecurityGroups"),
            "key-pair": ("describe_key_pairs", "KeyPairIds", "KeyPairs"),
            "network-interface": (
                "describe_network_interfaces",
                "NetworkInterfaceIds",
                "NetworkInterfaces",
            ),
        }
        if kind not in describe:
            return True  # a type this check does not know: assume it is still there
        method, param, key = describe[kind]
        try:
            found = getattr(self._ec2, method)(**{param: [ident]})
        except ClientError as exc:
            if _not_found(exc):
                return False
            raise
        return bool(found[key])

    def _budgets(self) -> Any:
        return self._client("budgets", region="us-east-1")  # global service endpoint

    def budget_exists(self, account: str, name: str) -> bool:
        try:
            self._budgets().describe_budget(AccountId=account, BudgetName=name)
        except ClientError as exc:
            if _not_found(exc):
                return False
            raise
        return True

    def create_budget(self, account: str, spec: BudgetSpec) -> None:
        subscribers = [{"SubscriptionType": "EMAIL", "Address": spec.email}]
        notifications = [
            {
                "Notification": {
                    "NotificationType": kind,
                    "ComparisonOperator": "GREATER_THAN",
                    "Threshold": pct,
                    "ThresholdType": "PERCENTAGE",
                },
                "Subscribers": subscribers,
            }
            for kind, levels in (
                ("ACTUAL", spec.actual_thresholds_pct),
                ("FORECASTED", spec.forecast_thresholds_pct),
            )
            for pct in levels
        ]
        self._budgets().create_budget(
            AccountId=account,
            Budget={
                "BudgetName": spec.name,
                "BudgetLimit": {"Amount": f"{spec.amount_usd:.2f}", "Unit": "USD"},
                "TimeUnit": "MONTHLY",
                "BudgetType": "COST",
                # Usage before credits: with promotional credits the net cost stays near zero,
                # and a budget that includes them would never alert while credits burn.
                "CostTypes": {"IncludeCredit": False, "IncludeRefund": False},
            },
            NotificationsWithSubscribers=notifications,
            ResourceTags=_tag_list(spec.tags),
        )
