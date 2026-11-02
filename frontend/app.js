const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  config: null,
  currentThread: null,
  currentConversation: null,
  currentRun: null,
  turns: [],
  pollTimer: null,
  pauseRequested: false,
  segmentActive: false,
  autoContinuing: false,
  report: "",
  history: loadStoredHistory(),
};

const statusLabels = {
  planning: "规划中",
  running: "执行中",
  paused: "已暂停",
  completed: "已完成",
  failed: "执行失败",
  waiting_human: "等待确认",
};

const taskStatusLabels = {
  pending: "等待",
  running: "执行中",
  completed: "完成",
  failed: "失败",
  blocked: "已阻塞",
  waiting_human: "待确认",
};

document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  renderHistory();
  await Promise.all([loadConfig(), checkHealth()]);
});

function bindEvents() {
  $("#openSettings").addEventListener("click", openSettings);
  $("#closeSettings").addEventListener("click", closeSettings);
  $("#cancelSettings").addEventListener("click", closeSettings);
  $("#settingsForm").addEventListener("submit", saveConfig);
  $("#researchForm").addEventListener("submit", startResearch);
  $("#followupForm").addEventListener("submit", startFollowup);
  $("#feedbackForm").addEventListener("submit", resumeResearch);
  $("#newResearch").addEventListener("click", showComposer);
  $("#backToComposer").addEventListener("click", showComposer);
  $("#refreshRun").addEventListener("click", () => refreshRun(true));
  $("#pauseExecution").addEventListener("click", pauseExecution);
  $("#resumeExecution").addEventListener("click", resumeExecution);
  $("#copyThread").addEventListener("click", copyThreadId);
  $("#copyReport").addEventListener("click", copyReport);
  $("#downloadReport").addEventListener("click", downloadReport);
  $("#clearHistory").addEventListener("click", clearHistory);
  $$("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      $("#goalInput").value = button.dataset.prompt;
      $("#goalInput").focus();
    });
  });
  $$('[data-toggle-secret]').forEach((button) => {
    button.addEventListener("click", () => toggleSecret(button));
  });
  $("#settingsDialog").addEventListener("click", (event) => {
    if (event.target === $("#settingsDialog")) closeSettings();
  });
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error();
    $("#serviceStatus").className = "service-status online";
    $("#serviceStatus").innerHTML = "<i></i> 服务正常";
  } catch {
    $("#serviceStatus").className = "service-status offline";
    $("#serviceStatus").innerHTML = "<i></i> 服务离线";
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/config");
    if (!response.ok) throw new Error("无法读取配置");
    state.config = await response.json();
    fillConfigForm(state.config);
    updateConfigStatus(state.config);
    if (!state.config.openai_configured) {
      setTimeout(openSettings, 350);
    }
  } catch (error) {
    toast(error.message, true);
  }
}

function fillConfigForm(config) {
  $("#openaiBaseUrl").value = config.openai_base_url || "https://api.openai.com/v1";
  $("#smallModel").value = config.small_model || "";
  $("#largeModel").value = config.large_model || "";
  $("#structuredOutput").value = config.structured_output_method || "json_schema";
  $("#maxAgentContextChars").value = config.max_agent_context_chars || 16000;
  $("#softWatermark").value = config.context_soft_watermark ?? 0.70;
  $("#hardWatermark").value = config.context_hard_watermark ?? 0.85;
  $("#toolResultMaxChars").value = config.context_tool_result_max_chars || 2400;
  $("#summaryMaxChars").value = config.context_summary_max_chars || 1200;
  $("#toolsEnabled").checked = config.tools_enabled;
  $("#qualityGateEnabled").checked = config.research_quality_gate_enabled;
  $("#initialSearches").value = config.research_initial_searches || 5;
  $("#maxSearchesPerTask").value = config.research_max_searches_per_task || 10;
  $("#completionScore").value = config.research_completion_score ?? 80;
  $("#minUniqueSources").value = config.research_min_unique_sources || 5;
  $("#minIndependentDomains").value = config.research_min_independent_domains || 3;
  $("#followupBatchSize").value = config.research_followup_batch_size || 2;
  $("#dynamicRouting").checked = config.dynamic_model_routing;
  $("#langsmithTracing").checked = config.langsmith_tracing;
  $("#langsmithProject").value = config.langsmith_project || "research-agent-local";
  $("#maxConcurrentWorkers").value = config.max_concurrent_workers ?? 5;
  $("#maxTotalTasks").value = config.max_total_tasks ?? 12;
  $("#maxTotalSearches").value = config.max_total_searches ?? 100;
  $("#maxTotalInputTokens").value = config.max_total_input_tokens ?? 500000;
  $("#maxTotalOutputTokens").value = config.max_total_output_tokens ?? 200000;
  $("#maxTotalCostUsd").value = config.max_total_cost_usd ?? 0;
  $("#maxRunSeconds").value = config.max_run_seconds ?? 1200;
  $("#maxConflictTasks").value = config.max_conflict_tasks ?? 2;
  $("#taskMaxSearches").value = config.task_max_searches ?? 10;
  $("#taskMaxSources").value = config.task_max_sources ?? 12;
  $("#taskMaxInputTokens").value = config.task_max_input_tokens ?? 60000;
  $("#taskMaxOutputTokens").value = config.task_max_output_tokens ?? 20000;
  $("#taskMaxSeconds").value = config.task_max_seconds ?? 300;
  $("#taskMaxRetries").value = config.task_max_retries ?? 2;
  $("#smallInputPrice").value = config.small_input_cost_per_million ?? 0;
  $("#smallOutputPrice").value = config.small_output_cost_per_million ?? 0;
  $("#largeInputPrice").value = config.large_input_cost_per_million ?? 0;
  $("#largeOutputPrice").value = config.large_output_cost_per_million ?? 0;
  $("#planApprovalMode").value = config.plan_approval_mode || "auto";
  $("#approvalTaskThreshold").value = config.plan_approval_task_threshold ?? 8;
  $("#approvalTokenThreshold").value = config.plan_approval_token_threshold ?? 200000;
  $("#approvalCostThreshold").value = config.plan_approval_cost_threshold_usd ?? 5;
  $("#openaiApiKey").placeholder = config.openai_configured ? "已配置；留空表示保留" : "输入 API Key";
  $("#tavilyApiKey").placeholder = config.tavily_configured ? "已配置；留空表示保留" : "实时调研必须配置";
  $("#langsmithApiKey").placeholder = config.langsmith_configured ? "已配置；留空表示保留" : "仅启用 Trace 时需要";
  populateModelSelectors(config);
}

function populateModelSelectors(config) {
  const models = config.available_models || [config.small_model, config.large_model].filter(Boolean);
  const labels = new Map([
    [config.small_model, "轻量模型"],
    [config.large_model, "高能力模型"],
  ]);
  [$("#modelSelect"), $("#followupModel")].forEach((select) => {
    const current = select.value;
    select.innerHTML = '<option value="">自动路由（推荐）</option>' + [...new Set(models)].map((model) =>
      `<option value="${escapeAttribute(model)}">${escapeHtml(model)} · ${escapeHtml(labels.get(model) || "已配置模型")}</option>`
    ).join("");
    select.value = models.includes(current) ? current : "";
  });
}

function updateConfigStatus(config) {
  const node = $("#configStatus");
  if (config.openai_configured && config.tavily_configured && config.tools_enabled) {
    node.className = "config-status ready";
    node.innerHTML = `<i></i><span>模型与实时搜索已就绪 · ${escapeHtml(config.small_model)} / ${escapeHtml(config.large_model)}</span>`;
  } else if (config.openai_configured) {
    node.className = "config-status";
    node.innerHTML = "<i></i><span>模型已配置；请配置 Tavily 并启用外部工具</span>";
  } else {
    node.className = "config-status";
    node.innerHTML = "<i></i><span>需要填写 API Key 和两个模型名称</span>";
  }
}

function openSettings() {
  if (!$("#settingsDialog").open) $("#settingsDialog").showModal();
}

function closeSettings() {
  $("#settingsDialog").close();
}

async function saveConfig(event) {
  event.preventDefault();
  const button = $("#saveSettings");
  setButtonLoading(button, true, "正在验证…");
  const payload = {
    openai_base_url: $("#openaiBaseUrl").value.trim(),
    small_model: $("#smallModel").value.trim(),
    large_model: $("#largeModel").value.trim(),
    structured_output_method: $("#structuredOutput").value,
    max_agent_context_chars: Number($("#maxAgentContextChars").value),
    context_soft_watermark: Number($("#softWatermark").value),
    context_hard_watermark: Number($("#hardWatermark").value),
    context_tool_result_max_chars: Number($("#toolResultMaxChars").value),
    context_summary_max_chars: Number($("#summaryMaxChars").value),
    tools_enabled: $("#toolsEnabled").checked,
    research_quality_gate_enabled: $("#qualityGateEnabled").checked,
    research_initial_searches: Number($("#initialSearches").value),
    research_max_searches_per_task: Number($("#maxSearchesPerTask").value),
    research_completion_score: Number($("#completionScore").value),
    research_min_unique_sources: Number($("#minUniqueSources").value),
    research_min_independent_domains: Number($("#minIndependentDomains").value),
    research_followup_batch_size: Number($("#followupBatchSize").value),
    dynamic_model_routing: $("#dynamicRouting").checked,
    max_concurrent_workers: Number($("#maxConcurrentWorkers").value),
    max_total_tasks: Number($("#maxTotalTasks").value),
    max_total_searches: Number($("#maxTotalSearches").value),
    max_total_input_tokens: Number($("#maxTotalInputTokens").value),
    max_total_output_tokens: Number($("#maxTotalOutputTokens").value),
    max_total_cost_usd: Number($("#maxTotalCostUsd").value),
    max_run_seconds: Number($("#maxRunSeconds").value),
    max_conflict_tasks: Number($("#maxConflictTasks").value),
    task_max_searches: Number($("#taskMaxSearches").value),
    task_max_sources: Number($("#taskMaxSources").value),
    task_max_input_tokens: Number($("#taskMaxInputTokens").value),
    task_max_output_tokens: Number($("#taskMaxOutputTokens").value),
    task_max_seconds: Number($("#taskMaxSeconds").value),
    task_max_retries: Number($("#taskMaxRetries").value),
    small_input_cost_per_million: Number($("#smallInputPrice").value),
    small_output_cost_per_million: Number($("#smallOutputPrice").value),
    large_input_cost_per_million: Number($("#largeInputPrice").value),
    large_output_cost_per_million: Number($("#largeOutputPrice").value),
    plan_approval_mode: $("#planApprovalMode").value,
    plan_approval_task_threshold: Number($("#approvalTaskThreshold").value),
    plan_approval_token_threshold: Number($("#approvalTokenThreshold").value),
    plan_approval_cost_threshold_usd: Number($("#approvalCostThreshold").value),
    langsmith_tracing: $("#langsmithTracing").checked,
    langsmith_project: $("#langsmithProject").value.trim() || "research-agent-local",
  };
  const secrets = {
    openai_api_key: $("#openaiApiKey").value.trim(),
    tavily_api_key: $("#tavilyApiKey").value.trim(),
    langsmith_api_key: $("#langsmithApiKey").value.trim(),
  };
  Object.entries(secrets).forEach(([key, value]) => {
    if (value) payload[key] = value;
  });
  try {
    const response = await fetch("/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await readJson(response);
    if (!response.ok) throw new Error(apiError(data, "保存配置失败"));
    state.config = data;
    fillConfigForm(data);
    updateConfigStatus(data);
    ["openaiApiKey", "tavilyApiKey", "langsmithApiKey"].forEach((id) => { $(`#${id}`).value = ""; });
    closeSettings();
    toast("API 配置已保存，研究团队可以开始工作了");
  } catch (error) {
    toast(error.message, true);
  } finally {
    setButtonLoading(button, false, "保存配置");
  }
}

async function startResearch(event) {
  event.preventDefault();
  if (!state.config?.openai_configured) {
    toast("请先配置模型 API", true);
    openSettings();
    return;
  }
  if (state.config.require_external_evidence && (!state.config.tavily_configured || !state.config.tools_enabled)) {
    toast("实时调研必须配置 Tavily API Key 并启用外部工具", true);
    openSettings();
    return;
  }
  const goal = $("#goalInput").value.trim();
  const userId = $("#userIdInput").value.trim() || "local-user";
  if (!goal) return;
  const threadId = crypto.randomUUID();
  const conversationId = crypto.randomUUID();
  const selectedModel = $("#modelSelect").value || null;
  state.currentThread = threadId;
  state.currentConversation = conversationId;
  state.currentRun = null;
  state.turns = [];
  state.report = "";
  state.pauseRequested = false;
  showRun({ thread_id: threadId, conversation_id: conversationId, user_id: userId, user_goal: goal, selected_model: selectedModel, status: "planning", tasks: [], final_report: null, errors: [], human_requests: [], reasoning_steps: [], paused: false, can_resume: false }, goal);
  upsertHistory({ conversationId, latestThreadId: threadId, goal, status: "planning", updatedAt: Date.now() });
  await runSegmentLoop({ user_goal: goal, user_id: userId, thread_id: threadId, conversation_id: conversationId, model: selectedModel });
}

async function startFollowup(event) {
  event.preventDefault();
  if (!state.currentRun || state.currentRun.status !== "completed" || state.segmentActive) return;
  const message = $("#followupInput").value.trim();
  if (!message) return;
  const parentThreadId = state.currentThread;
  const threadId = crypto.randomUUID();
  const conversationId = state.currentConversation || state.currentRun.conversation_id || parentThreadId;
  const selectedModel = $("#followupModel").value || null;
  const userId = state.currentRun.user_id || "local-user";
  state.currentThread = threadId;
  state.pauseRequested = false;
  showRun({
    thread_id: threadId,
    conversation_id: conversationId,
    parent_thread_id: parentThreadId,
    user_id: userId,
    user_goal: message,
    selected_model: selectedModel,
    status: "planning",
    tasks: [],
    final_report: null,
    errors: [],
    human_requests: [],
    reasoning_steps: [],
    paused: false,
    can_resume: false,
  });
  $("#followupInput").value = "";
  await runSegmentLoop({
    user_goal: message,
    user_id: userId,
    thread_id: threadId,
    conversation_id: conversationId,
    parent_thread_id: parentThreadId,
    model: selectedModel,
  });
}

async function runSegmentLoop(payload) {
  if (state.segmentActive) return;
  let nextPayload = payload;
  try {
    while (nextPayload) {
      state.segmentActive = true;
      beginPolling(state.currentThread);
      updateExecutionButtons();
      const response = await fetch("/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(nextPayload),
      });
      const data = await readJson(response);
      state.segmentActive = false;
      if (!response.ok) throw new Error(apiError(data, "调研任务执行失败"));
      state.autoContinuing = Boolean(
        data.can_resume
        && !["failed", "completed", "waiting_human"].includes(data.status)
        && !state.pauseRequested
      );
      renderRun(data);
      if (!data.can_resume || ["failed", "completed", "waiting_human"].includes(data.status)) {
        state.autoContinuing = false;
        return;
      }
      if (state.pauseRequested) {
        state.autoContinuing = false;
        stopPolling();
        updateExecutionButtons();
        toast("任务已在安全 Checkpoint 暂停");
        return;
      }
      nextPayload = {
        resume: true,
        thread_id: data.thread_id,
        user_id: data.user_id || "local-user",
      };
    }
  } catch (error) {
    stopPolling();
    renderRequestFailure(error.message);
    if (error.message.includes("model") || error.message.includes("API")) openSettings();
  } finally {
    state.segmentActive = false;
    state.autoContinuing = false;
    updateExecutionButtons();
  }
}

function pauseExecution() {
  state.pauseRequested = true;
  updateExecutionButtons();
  toast(state.segmentActive ? "已请求暂停，当前 Agent 批次结束后生效" : "任务已暂停");
}

async function resumeExecution() {
  if (!state.currentRun?.can_resume || state.segmentActive) return;
  state.pauseRequested = false;
  $("#humanPanel").hidden = true;
  await runSegmentLoop({
    resume: true,
    thread_id: state.currentThread,
    user_id: state.currentRun.user_id || "local-user",
  });
}

function beginPolling(threadId) {
  stopPolling();
  state.pollTimer = setInterval(() => refreshRun(false, threadId), 1400);
}

function stopPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
}

async function refreshRun(showMessage = false, threadId = state.currentThread) {
  if (!threadId) return;
  try {
    const response = await fetch(`/research/${encodeURIComponent(threadId)}`);
    if (response.status === 404) return;
    const data = await readJson(response);
    if (!response.ok) throw new Error(apiError(data, "读取任务失败"));
    if (threadId === state.currentThread) renderRun(data);
    if (showMessage) toast("已刷新最新进度");
  } catch (error) {
    if (showMessage) toast(error.message, true);
  }
}

function showRun(run, goal) {
  $("#composerView").hidden = true;
  $("#runView").hidden = false;
  state.currentConversation = run.conversation_id || state.currentConversation || run.thread_id;
  $("#runGoal").textContent = goal || historyGoal(state.currentConversation) || "研究对话";
  $("#threadId").textContent = run.thread_id;
  renderRun(run);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderRun(run) {
  state.currentRun = run;
  state.currentThread = run.thread_id;
  state.currentConversation = run.conversation_id || state.currentConversation || run.thread_id;
  state.report = run.final_report || "";
  const status = run.status || "planning";
  const tasks = run.tasks || [];
  const completed = tasks.filter((task) => task.status === "completed").length;
  const progress = tasks.length ? Math.round((completed / tasks.length) * 100) : 0;
  const evidences = run.evidences?.length || 0;
  const active = tasks.find((task) => task.status === "running" || task.status === "waiting_human");

  const conversationTitle = historyGoal(state.currentConversation);
  if (conversationTitle) $("#runGoal").textContent = conversationTitle;
  else if (run.user_goal) $("#runGoal").textContent = run.user_goal;
  $("#runStatusBadge").className = `status-badge status-${status}`;
  $("#runStatusBadge").textContent = statusLabels[status] || status;
  $("#threadId").textContent = run.thread_id;
  $("#progressPercent").textContent = `${progress}%`;
  $("#progressBar").style.width = `${progress}%`;
  $("#taskCounter").textContent = `${completed} / ${tasks.length} 节点`;
  $("#completedCount").textContent = completed;
  $("#evidenceCount").textContent = evidences;
  $("#errorCount").textContent = (run.errors || []).length;
  $("#activeAgent").textContent = active ? `${active.assigned_agent} · ${active.task_id}` : status === "completed" ? "全部完成" : "等待调度";
  $("#progressText").textContent = progressMessage(status, active, tasks);
  renderTasks(tasks);
  upsertTurnFromRun(run);
  renderConversation();
  renderReasoning(run.reasoning_steps || [], run);
  renderContextCompressions(run.context_compressions || []);
  renderBudget(run);
  renderErrors(run.errors || []);
  renderProvenance(run.tool_calls || [], run.evidences || []);
  renderHumanPanel(run);
  renderReport(run.final_report);
  $("#followupPanel").hidden = status !== "completed";
  $("#downloadReport").disabled = !run.final_report;
  updateExecutionButtons();

  upsertHistory({
    conversationId: state.currentConversation,
    latestThreadId: run.thread_id,
    goal: $("#runGoal").textContent,
    status,
    updatedAt: Date.now(),
  });
  if (["completed", "failed", "waiting_human"].includes(status) || (run.paused && state.pauseRequested)) stopPolling();
}

function renderBudget(run) {
  const budget = run.budget || {};
  const limits = budget.limits || {};
  const lead = run.lead_decision || {};
  const estimate = run.plan_estimate;
  const metrics = run.metrics || {};
  $("#leadMode").textContent = lead.mode
    ? `${lead.mode === "multi" ? "Multi-Agent" : "Single-Agent"} · ${lead.recommended_workers || 1} Worker`
    : "等待规划";
  $("#budgetWorkers").textContent = `${budget.peak_workers || 0} / ${limits.max_concurrent_workers || 0}`;
  $("#budgetSearches").textContent = `${budget.searches || 0} / ${limits.max_searches || 0}`;
  $("#budgetTokens").textContent = Number((budget.input_tokens || 0) + (budget.output_tokens || 0)).toLocaleString("zh-CN");
  $("#budgetCost").textContent = `$${Number(budget.estimated_cost_usd || 0).toFixed(4)}`;
  $("#budgetElapsed").textContent = `${(Number(budget.elapsed_ms || 0) / 1000).toFixed(1)}s`;
  const estimateNode = $("#planEstimate");
  if (budget.exhausted_reason) {
    estimateNode.className = "plan-estimate exhausted";
    estimateNode.textContent = `预算断路：${budget.exhausted_reason}`;
  } else if (estimate) {
    estimateNode.className = "plan-estimate";
    const quality = metrics.claim_support_rate == null
      ? ""
      : ` 当前 Claim 支持率 ${Math.round(metrics.claim_support_rate * 100)}%，引用落库率 ${Math.round((metrics.citation_grounding_rate || 0) * 100)}%，独立域名 ${metrics.independent_domains || 0}。`;
    estimateNode.textContent = `执行前估算：${estimate.task_count} 个任务、${estimate.estimated_searches} 次搜索、${Number(estimate.estimated_input_tokens + estimate.estimated_output_tokens).toLocaleString("zh-CN")} tokens、$${Number(estimate.estimated_cost_usd || 0).toFixed(4)}；${lead.reason || ""}${quality}`;
  } else {
    estimateNode.className = "plan-estimate";
    estimateNode.textContent = "Planner 完成后显示执行前估算。";
  }
}

function updateExecutionButtons() {
  const run = state.currentRun;
  const resumable = Boolean(run?.can_resume);
  const activelyExecuting = state.segmentActive || state.autoContinuing;
  $("#pauseExecution").hidden = !resumable || state.pauseRequested || !activelyExecuting;
  $("#pauseExecution").disabled = !activelyExecuting;
  $("#resumeExecution").hidden = !resumable || activelyExecuting || (!state.pauseRequested && !run?.paused);
  $("#resumeExecution").disabled = activelyExecuting;
}

function renderProvenance(toolCalls, evidences) {
  const toolList = $("#toolCallList");
  const evidenceList = $("#evidenceList");
  toolList.innerHTML = toolCalls.length ? toolCalls.map((call) => {
    const query = call.arguments?.query || call.arguments?.sql || call.purpose || "已执行外部调用";
    const duration = Number.isFinite(call.duration_ms) ? `${Math.round(call.duration_ms)} ms` : "执行中";
    return `<article class="tool-call-item ${escapeHtml(call.status)}">
      <div><strong>${escapeHtml(call.tool_name)}</strong><span>${escapeHtml(call.status)}</span></div>
      <p>${escapeHtml(query)}</p><small>${escapeHtml(call.task_id)} · ${call.evidence_count} 条证据 · ${duration}</small>
      ${call.error ? `<em>${escapeHtml(call.error)}</em>` : ""}
    </article>`;
  }).join("") : '<p class="empty-provenance">等待 Researcher 自主调用检索工具…</p>';

  evidenceList.innerHTML = evidences.length ? evidences.map((evidence) => {
    const url = safeExternalUrl(evidence.source_url);
    const title = evidence.title || evidence.source_url;
    const score = Math.round((evidence.confidence || 0) * 100);
    const heading = url
      ? `<a href="${escapeAttribute(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(title)} ↗</a>`
      : `<strong>${escapeHtml(title)}</strong>`;
    return `<article class="evidence-item">${heading}<p>${escapeHtml(evidence.snippet)}</p><small>${escapeHtml(evidence.tool_name)} · 相关度 ${score}% · ${escapeHtml(evidence.evidence_id)}</small></article>`;
  }).join("") : '<p class="empty-provenance">成功检索后将在这里显示真实来源 URL。</p>';
}

function progressMessage(status, active, tasks) {
  if (status === "paused") return "已保存 Checkpoint，可随时继续执行且不会重跑已完成节点。";
  if (status === "completed") return "调研完成，结构化报告已生成。";
  if (status === "failed") return "执行遇到无法自动恢复的异常。";
  if (status === "waiting_human") return "系统已暂停，等待你的确认。";
  if (active) return `${active.title} · ${taskStatusLabels[active.status] || active.status}`;
  if (!tasks.length) return "正在解析目标并生成动态任务 DAG…";
  return "Supervisor 正在安排下一批 Agent 节点…";
}

function renderTasks(tasks) {
  const list = $("#taskList");
  if (!tasks.length) {
    list.innerHTML = '<div class="task-skeleton"></div><div class="task-skeleton"></div><div class="task-skeleton"></div>';
    return;
  }
  list.innerHTML = tasks.map((task, index) => {
    const dependency = task.dependencies?.length ? `依赖 ${task.dependencies.join(", ")}` : "可独立执行";
    const detail = task.error || task.description;
    const quality = Number.isFinite(task.quality_score)
      ? `<span>· 质量 ${task.quality_score}/100</span><span>· Tavily ${task.search_count || 0} 次</span>`
      : "";
    const budget = task.budget
      ? `<span>· 预算 搜索 ${task.budget.max_searches} / 来源 ${task.budget.max_sources} / ${task.budget.max_seconds}s</span>`
      : "";
    return `<article class="task-item ${escapeHtml(task.status)}">
      <span class="task-index">${task.status === "completed" ? "✓" : String(index + 1).padStart(2, "0")}</span>
      <div class="task-copy"><strong>${escapeHtml(task.title)}</strong><p>${escapeHtml(detail)}</p><div class="task-meta"><span>${escapeHtml(dependency)}</span><span>· 尝试 ${task.attempts}</span>${quality}${budget}</div></div>
      <span class="agent-pill">${escapeHtml(task.assigned_agent)} · ${escapeHtml(taskStatusLabels[task.status] || task.status)}</span>
    </article>`;
  }).join("");
}

function upsertTurnFromRun(run) {
  if (!run.thread_id) return;
  const turn = {
    threadId: run.thread_id,
    parentThreadId: run.parent_thread_id || null,
    user: run.user_goal || "",
    assistant: run.final_report || "",
    status: run.status || "planning",
    model: run.selected_model || "",
  };
  const index = state.turns.findIndex((item) => item.threadId === turn.threadId);
  if (index >= 0) state.turns[index] = { ...state.turns[index], ...turn };
  else state.turns.push(turn);
}

function renderConversation() {
  const node = $("#conversationMessages");
  $("#turnCounter").textContent = `共 ${state.turns.length || 1} 轮`;
  if (!state.turns.length) {
    node.innerHTML = '<p class="empty-provenance">发送问题后，对话会显示在这里。</p>';
    return;
  }
  node.innerHTML = state.turns.map((turn) => {
    const model = turn.model || "自动模型路由";
    const assistant = turn.assistant
      ? `<div class="message-markdown">${markdownToSafeHtml(turn.assistant)}</div>`
      : `<div class="message-pending"><i></i>${turn.status === "failed" ? "本轮执行失败，请查看异常记录" : "研究团队正在规划、检索与验证…"}</div>`;
    return `<div class="conversation-turn" data-thread="${escapeAttribute(turn.threadId)}">
      <article class="message message-user"><span>你</span><div><p>${escapeHtml(turn.user)}</p><small>${escapeHtml(model)}</small></div></article>
      <article class="message message-assistant"><span>AI</span><div>${assistant}</div></article>
    </div>`;
  }).join("");
}

const reasoningLabels = {
  requirement_parsed: "理解并结构化用户需求",
  plan_created: "生成研究计划与任务依赖",
  tasks_dispatched: "Supervisor 分派任务",
  task_completed: "Agent 完成研究节点",
  context_hard_compressed: "硬水位压缩并保存上下文快照",
  workflow_completed: "交叉验证完成并生成报告",
  workflow_failed: "工作流终止",
  human_intervention: "等待人工确认",
  "model.invoke": "调用模型进行结构化推理",
  "worker.completed": "Agent 节点完成",
  "worker.failed": "Agent 节点失败",
  "worker.replan": "Verifier 要求补充研究",
  "worker.waiting_human": "节点等待人工确认",
};

function renderReasoning(steps, run) {
  const node = $("#reasoningTimeline");
  if (!steps.length) {
    node.innerHTML = '<p class="empty-provenance">正在等待 Planner 生成第一步计划…</p>';
    return;
  }
  node.innerHTML = steps.map((step) => {
    const title = reasoningLabels[step.name] || step.name.replaceAll("_", " ");
    const model = step.kind === "model"
      ? `${run.selected_model || `${step.model_tier || "auto"} tier`} · ${step.schema_name || "structured output"}`
      : [step.agent, step.task_id].filter(Boolean).join(" · ");
    const tokens = step.kind === "model" ? ` · ${step.input_tokens || 0} → ${step.output_tokens || 0} tokens` : "";
    const duration = Number.isFinite(step.duration_ms) ? ` · ${Math.round(step.duration_ms)} ms` : "";
    return `<article class="reasoning-step ${escapeHtml(step.status)}">
      <i></i><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(step.detail || model || "已记录安全执行摘要")}</p><small>${escapeHtml(step.kind)}${tokens}${duration}</small></div>
    </article>`;
  }).join("");
}

function renderContextCompressions(events) {
  const saved = events.reduce((total, event) => total + (event.saved_chars || 0), 0);
  const hard = events.filter((event) => event.trigger === "hard").length;
  const references = events.reduce(
    (total, event) => total + (event.references_preserved || 0),
    0,
  );
  $("#compressionCount").textContent = events.length;
  $("#compressionSaved").textContent = saved.toLocaleString("zh-CN");
  $("#compressionHardCount").textContent = hard;
  $("#compressionReferences").textContent = references;
  const node = $("#compressionList");
  if (!events.length) {
    node.innerHTML = '<p class="empty-provenance">尚未达到上下文软水位。</p>';
    return;
  }
  node.innerHTML = [...events].reverse().map((event) => {
    const savedRate = Math.round((1 - (event.compression_ratio || 0)) * 100);
    const checkpoint = event.checkpoint_required ? " · 已要求 Checkpoint" : "";
    return `<article class="compression-item ${escapeHtml(event.trigger)}">
      <div><strong>第 ${event.applied_level} 层 · ${event.trigger === "hard" ? "硬水位" : "软水位"}</strong><span>${savedRate}%</span></div>
      <p>${Number(event.original_chars || 0).toLocaleString("zh-CN")} → ${Number(event.compressed_chars || 0).toLocaleString("zh-CN")} 字符 · 保留 ${event.references_preserved || 0} 个引用${checkpoint}</p>
      <small>${escapeHtml(event.context_id)} · ${escapeHtml(event.boundary)}</small>
    </article>`;
  }).join("");
}

function renderErrors(errors) {
  const box = $("#errorBox");
  if (!errors.length) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.innerHTML = errors.slice(-3).map((error) => `<div><strong>${escapeHtml(error.code)}</strong> · ${escapeHtml(error.message)}</div>`).join("");
}

function renderHumanPanel(run) {
  const panel = $("#humanPanel");
  const requests = run.human_requests || [];
  const waitingForApproval = run.status === "waiting_human";
  const userPaused = run.status === "paused" && !state.autoContinuing;
  panel.hidden = !waitingForApproval && !userPaused;
  if (panel.hidden) return;
  $("#feedbackTitle").textContent = waitingForApproval ? "需要你的确认" : "调整本次调研";
  $("#feedbackDescription").textContent = waitingForApproval
    ? "系统已安全暂停。批准操作或修改研究要求后继续。"
    : "任务已停在安全 Checkpoint。你可以修改目标、增加约束，再从受影响节点继续执行。";
  $("#feedbackSubmit").textContent = waitingForApproval ? "确认并继续执行 →" : "应用新要求并继续 →";
  $("#humanRequests").innerHTML = requests.length ? requests.map((request) => {
    const approval = request.approval_token
      ? `<label><input type="checkbox" name="approval" value="${escapeAttribute(request.approval_token)}" checked /> 批准此操作</label>`
      : "<span>请修改要求后继续</span>";
    return `<div class="human-request"><span><strong>${escapeHtml(request.task_id)}</strong> · ${escapeHtml(request.reason)}</span>${approval}</div>`;
  }).join("") : '<div class="human-request"><span>已保存当前进度；不填写受影响节点时，Planner 会自动分析修改范围。</span></div>';
}

async function resumeResearch(event) {
  event.preventDefault();
  if (!state.currentThread || !state.currentRun) return;
  const approvals = $$('input[name="approval"]:checked').map((input) => input.value);
  const constraints = splitValues($("#additionalConstraints").value);
  const affected = splitValues($("#affectedTasks").value);
  const revisedGoal = $("#revisedGoal").value.trim();
  const feedback = { approvals, additional_constraints: constraints, affected_task_ids: affected };
  if (revisedGoal) feedback.revised_goal = revisedGoal;
  const button = $("#feedbackSubmit");
  setButtonLoading(button, true, "正在应用并重新规划…");
  $("#humanPanel").hidden = true;
  state.pauseRequested = false;
  try {
    await runSegmentLoop({
      resume: true,
      thread_id: state.currentThread,
      user_id: state.currentRun.user_id || "local-user",
      human_feedback: feedback,
    });
    $("#feedbackForm").reset();
  } finally {
    setButtonLoading(button, false, "应用新要求并继续 →");
  }
}

function renderReport(report) {
  const panel = $("#reportPanel");
  panel.hidden = !report;
  if (!report) return;
  $("#reportContent").innerHTML = markdownToSafeHtml(report);
}

function markdownToSafeHtml(markdown) {
  const lines = escapeHtml(markdown).split("\n");
  let inList = false;
  const html = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("- ")) {
      if (!inList) { html.push("<ul>"); inList = true; }
      html.push(`<li>${formatInline(line.slice(2))}</li>`);
      continue;
    }
    if (inList) { html.push("</ul>"); inList = false; }
    if (!line) { html.push(""); continue; }
    if (line.startsWith("### ")) html.push(`<h3>${formatInline(line.slice(4))}</h3>`);
    else if (line.startsWith("## ")) html.push(`<h2>${formatInline(line.slice(3))}</h2>`);
    else if (line.startsWith("# ")) html.push(`<h1>${formatInline(line.slice(2))}</h1>`);
    else html.push(`<p>${formatInline(line)}</p>`);
  }
  if (inList) html.push("</ul>");
  return html.join("\n");
}

function formatInline(value) {
  return value
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/\[(evidence:[^\]]+)\]/g, (match, evidenceId) => {
      const evidence = state.currentRun?.evidences?.find((item) => item.evidence_id === evidenceId);
      const url = evidence ? safeExternalUrl(evidence.source_url) : null;
      return url ? `<a href="${escapeAttribute(url)}" target="_blank" rel="noopener noreferrer">${match} ↗</a>` : match;
    });
}

function safeExternalUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function showComposer() {
  $("#runView").hidden = true;
  $("#composerView").hidden = false;
  stopPolling();
  state.currentThread = null;
  state.currentConversation = null;
  state.currentRun = null;
  state.turns = [];
  state.pauseRequested = false;
  state.autoContinuing = false;
  $$(".history-item").forEach((item) => item.classList.remove("active"));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderRequestFailure(message) {
  const run = state.currentRun || { thread_id: state.currentThread, tasks: [], errors: [], human_requests: [] };
  run.status = "failed";
  run.errors = [...(run.errors || []), { code: "REQUEST_FAILED", message }];
  renderRun(run);
  toast(message, true);
}

function upsertHistory(entry) {
  const conversationId = entry.conversationId || entry.threadId;
  const latestThreadId = entry.latestThreadId || entry.threadId;
  if (!conversationId || !latestThreadId) return;
  const normalized = { ...entry, conversationId, latestThreadId };
  delete normalized.threadId;
  const existing = state.history.find((item) => item.conversationId === conversationId);
  state.history = [
    { ...(existing || {}), ...normalized },
    ...state.history.filter((item) => item.conversationId !== conversationId),
  ].slice(0, 12);
  localStorage.setItem("deepresearch.history", JSON.stringify(state.history));
  renderHistory();
}

function renderHistory() {
  const node = $("#historyList");
  if (!state.history.length) {
    node.innerHTML = '<p class="empty-sidebar">暂无历史任务</p>';
    return;
  }
  node.innerHTML = state.history.map((item) => `<button class="history-item ${item.conversationId === state.currentConversation ? "active" : ""}" data-thread="${escapeAttribute(item.latestThreadId)}" data-conversation="${escapeAttribute(item.conversationId)}" data-status="${escapeAttribute(item.status || "planning")}" type="button"><strong>${escapeHtml(item.goal || "未命名研究对话")}</strong><span><i></i>${escapeHtml(statusLabels[item.status] || item.status || "进行中")} · ${formatTime(item.updatedAt)}</span></button>`).join("");
  $$(".history-item").forEach((button) => {
    button.addEventListener("click", async () => {
      const item = state.history.find((entry) => entry.conversationId === button.dataset.conversation);
      if (!item) return;
      state.currentThread = item.latestThreadId;
      state.currentConversation = item.conversationId;
      state.turns = [];
      showRun({ thread_id: item.latestThreadId, conversation_id: item.conversationId, user_id: "local-user", status: item.status, tasks: [], errors: [], human_requests: [], reasoning_steps: [] }, item.goal);
      await loadConversation(item.latestThreadId, true);
    });
  });
}

function loadStoredHistory() {
  try {
    return JSON.parse(localStorage.getItem("deepresearch.history") || "[]").map((item) => ({
      ...item,
      conversationId: item.conversationId || item.threadId,
      latestThreadId: item.latestThreadId || item.threadId,
    }));
  }
  catch { return []; }
}

async function loadConversation(latestThreadId, showMessage = false) {
  try {
    const runs = [];
    const visited = new Set();
    let threadId = latestThreadId;
    while (threadId && !visited.has(threadId) && runs.length < 30) {
      visited.add(threadId);
      const response = await fetch(`/research/${encodeURIComponent(threadId)}`);
      const data = await readJson(response);
      if (!response.ok) throw new Error(apiError(data, "读取对话失败"));
      runs.push(data);
      threadId = data.parent_thread_id;
    }
    runs.reverse();
    state.turns = runs.map((run) => ({
      threadId: run.thread_id,
      parentThreadId: run.parent_thread_id || null,
      user: run.user_goal || "",
      assistant: run.final_report || "",
      status: run.status || "planning",
      model: run.selected_model || "",
    }));
    const latest = runs[runs.length - 1];
    if (latest) renderRun(latest);
    if (showMessage) toast(`已恢复 ${runs.length} 轮研究对话`);
  } catch (error) {
    if (showMessage) toast(error.message, true);
  }
}

function clearHistory() {
  state.history = [];
  localStorage.removeItem("deepresearch.history");
  renderHistory();
  toast("本地任务历史已清空");
}

function historyGoal(threadId) {
  return state.history.find((item) =>
    item.conversationId === threadId || item.latestThreadId === threadId
  )?.goal;
}

async function copyThreadId() {
  if (!state.currentThread) return;
  await navigator.clipboard.writeText(state.currentThread);
  toast("任务 ID 已复制");
}

async function copyReport() {
  if (!state.report) return;
  await navigator.clipboard.writeText(state.report);
  toast("报告全文已复制");
}

function downloadReport() {
  if (!state.report) return;
  const blob = new Blob([state.report], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `research-report-${state.currentThread?.slice(0, 8) || "result"}.md`;
  link.click();
  URL.revokeObjectURL(url);
}

function toggleSecret(button) {
  const input = $(`#${button.dataset.toggleSecret}`);
  input.type = input.type === "password" ? "text" : "password";
  button.textContent = input.type === "password" ? "显示" : "隐藏";
}

function splitValues(value) {
  return value.split(/[,，\n]/).map((item) => item.trim()).filter(Boolean);
}

function setButtonLoading(button, loading, text) {
  button.disabled = loading;
  button.textContent = text;
}

function toast(message, isError = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${isError ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.className = "toast"; }, 3200);
}

async function readJson(response) {
  try { return await response.json(); }
  catch { return {}; }
}

function apiError(data, fallback) {
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map((item) => item.msg).join("；");
  return fallback;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/`/g, "&#096;");
}

function formatTime(timestamp) {
  if (!timestamp) return "刚刚";
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(timestamp));
}
