param(
    [string]$ProjectId = "tannu-intellidoc-20260912",
    [string]$Region = "asia-south1",
    [string]$Instance = "tannu-intellidoc-db",
    [string]$Database = "intellidoc",
    [string]$DatabaseUser = "intellidoc",
    [string]$BackendService = "tannu-intellidoc-backend"
)

$ErrorActionPreference = "Stop"
$projectNumber = (gcloud projects describe $ProjectId --format="value(projectNumber)").Trim()
$runtimeServiceAccount = "$projectNumber-compute@developer.gserviceaccount.com"
$databasePassword = [Guid]::NewGuid().ToString("N") + [Guid]::NewGuid().ToString("N")
$secretFile = New-TemporaryFile
$secretFilePath = $secretFile.FullName

try {
    $databaseExists = gcloud sql databases list `
        --instance=$Instance `
        --project=$ProjectId `
        --filter="name=$Database" `
        --format="value(name)"
    if (-not $databaseExists) {
        gcloud sql databases create $Database --instance=$Instance --project=$ProjectId --quiet
        if ($LASTEXITCODE -ne 0) { throw "Database creation failed." }
    }

    $userExists = gcloud sql users list `
        --instance=$Instance `
        --project=$ProjectId `
        --filter="name=$DatabaseUser" `
        --format="value(name)"
    if ($userExists) {
        gcloud sql users set-password $DatabaseUser --instance=$Instance --project=$ProjectId --password=$databasePassword
    }
    else {
        gcloud sql users create $DatabaseUser --instance=$Instance --project=$ProjectId --password=$databasePassword
    }
    if ($LASTEXITCODE -ne 0) { throw "Database user configuration failed." }

    $connectionName = (gcloud sql instances describe $Instance --project=$ProjectId --format="value(connectionName)").Trim()
    $databaseUrl = "postgresql+psycopg2://${DatabaseUser}:${databasePassword}@/$Database`?host=/cloudsql/$connectionName"
    [IO.File]::WriteAllText(
        $secretFilePath,
        $databaseUrl,
        [Text.UTF8Encoding]::new($false)
    )

    $secretExists = gcloud secrets describe intellidoc-database-url `
        --project=$ProjectId `
        --format="value(name)" 2>$null
    if ($secretExists) {
        gcloud secrets versions add intellidoc-database-url --project=$ProjectId --data-file=$secretFilePath
    }
    else {
        gcloud secrets create intellidoc-database-url --project=$ProjectId --data-file=$secretFilePath
    }
    if ($LASTEXITCODE -ne 0) { throw "Database secret creation failed." }

    foreach ($role in @("roles/cloudsql.client", "roles/secretmanager.secretAccessor")) {
        gcloud projects add-iam-policy-binding $ProjectId `
            --member="serviceAccount:$runtimeServiceAccount" `
            --role=$role `
            --condition=None `
            --quiet | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "IAM configuration failed for $role." }
    }

    $service = gcloud run services describe $BackendService `
        --project=$ProjectId `
        --region=$Region `
        --format=json | ConvertFrom-Json
    $databaseEnvironment = $service.spec.template.spec.containers[0].env |
        Where-Object { $_.name -eq "DATABASE_URL" }
    if ($databaseEnvironment.value) {
        gcloud run services update $BackendService `
            --project=$ProjectId `
            --region=$Region `
            --remove-env-vars=DATABASE_URL `
            --quiet
        if ($LASTEXITCODE -ne 0) { throw "Old database variable removal failed." }
    }

    gcloud run services update $BackendService `
        --project=$ProjectId `
        --region=$Region `
        --add-cloudsql-instances=$connectionName `
        --set-secrets="DATABASE_URL=intellidoc-database-url:latest" `
        --quiet
    if ($LASTEXITCODE -ne 0) { throw "Backend Cloud SQL connection failed." }

    Write-Host "Cloud SQL connected: $connectionName"
}
finally {
    $databasePassword = $null
    if (Test-Path -LiteralPath $secretFilePath) {
        Remove-Item -LiteralPath $secretFilePath -Force
    }
}
