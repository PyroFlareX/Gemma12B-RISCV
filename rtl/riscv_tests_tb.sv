`timescale 1ns/1ps
// Runs one riscv-tests rv32ui image (+hex=<file>) on rv32i_core with unified memory.
// PASS: ECALL trap with a0==0 (a7==93 exit convention). Any other trap or timeout fails.
module riscv_tests_tb;
    localparam int WORDS = 65536; // 256 KiB, accommodates ACT self-check images
    // HALT=1: legacy ECALL-exit images. HALT=0 (-PHALT=0): official rv32mi/rv32ui "p" env, pass/fail via tohost store.
    parameter bit HALT = 1'b1;
    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    logic [31:0] mem [0:WORDS-1];
    logic imem_valid, imem_ready, imem_fault, dmem_valid, dmem_write, dmem_ready, dmem_fault;
    logic [31:0] imem_addr, imem_rdata, dmem_addr, dmem_wdata, dmem_rdata;
    logic [3:0] dmem_wstrb;
    logic halted, trap_valid, retire_valid;
    logic [31:0] trap_pc, trap_value, retire_pc, retire_insn, retire_next_pc;
    logic [3:0] trap_cause;

    assign imem_ready = imem_valid;
    assign imem_fault = imem_addr[31:2] >= WORDS;
    assign imem_rdata = imem_fault ? 32'd0 : mem[imem_addr[17:2]];
    assign dmem_ready = dmem_valid;
    assign dmem_fault = dmem_addr[31:2] >= WORDS;
    assign dmem_rdata = dmem_fault ? 32'd0 : mem[dmem_addr[17:2]];

    rv32i_core #(.HALT_ON_TRAP(HALT)) dut (.clk, .rst_n, .imem_valid, .imem_addr, .imem_ready, .imem_rdata, .imem_fault,
        .dmem_valid, .dmem_write, .dmem_addr, .dmem_wdata, .dmem_wstrb, .dmem_ready, .dmem_rdata,
        .dmem_fault, .halted, .trap_valid, .trap_pc, .trap_cause, .trap_value,
        .retire_valid, .retire_pc, .retire_insn, .retire_next_pc);

    always_ff @(posedge clk)
        if (rst_n && dmem_valid && dmem_write && !dmem_fault) begin
            if (dmem_wstrb[0]) mem[dmem_addr[17:2]][7:0]   <= dmem_wdata[7:0];
            if (dmem_wstrb[1]) mem[dmem_addr[17:2]][15:8]  <= dmem_wdata[15:8];
            if (dmem_wstrb[2]) mem[dmem_addr[17:2]][23:16] <= dmem_wdata[23:16];
            if (dmem_wstrb[3]) mem[dmem_addr[17:2]][31:24] <= dmem_wdata[31:24];
        end

    string hexfile, memdumpfile, sigreffile;
    integer tohost_addr;
    logic [31:0] tohost_val;
    logic tohost_hit;
    // ACT (riscv-arch-test) signature comparison: the framework's verdict is the
    // signature region [sigbegin,sigend) matching the Sail reference, not the tohost
    // value (HTIF writes the summary string to the tohost address).
    integer sigbegin, sigend, sigwords, sigmism;
    logic [31:0] sigref [0:4095];
    always_ff @(posedge clk)
        if (rst_n && dmem_valid && dmem_write && dmem_addr == tohost_addr[31:0] && dmem_wstrb[0]) begin
            tohost_val <= dmem_wdata; tohost_hit <= 1'b1;
        end
    integer k, cycles, expect_cause;
    initial begin
        for (k = 0; k < WORDS; k = k + 1) mem[k] = 32'd0;
        if (!$value$plusargs("hex=%s", hexfile)) $fatal(1, "need +hex=");
        if (!$value$plusargs("expect=%d", expect_cause)) expect_cause = -1;
        $readmemh(hexfile, mem);
        tohost_hit = 0; tohost_addr = -1;
        void'($value$plusargs("tohost=%h", tohost_addr));
        if (!$value$plusargs("memdump=%s", memdumpfile)) memdumpfile = "";
        sigbegin = -1; sigend = -1; sigreffile = "";
        void'($value$plusargs("sigbegin=%h", sigbegin));
        void'($value$plusargs("sigend=%h", sigend));
        void'($value$plusargs("sigref=%s", sigreffile));
        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1;
        cycles = 0;
        while (!halted && !tohost_hit && cycles < 2000000) begin @(posedge clk); cycles = cycles + 1; end
        if (memdumpfile != "") $writememh(memdumpfile, mem);
        #1;
        if (sigreffile != "" && sigbegin >= 0 && sigend > sigbegin) begin
            $readmemh(sigreffile, sigref);
            sigwords = (sigend - sigbegin) / 4;
            sigmism = 0;
            for (k = 0; k < sigwords; k = k + 1)
                if (mem[sigbegin/4 + k] !== sigref[k]) sigmism = sigmism + 1;
            if (sigmism == 0) $display("PASS (signature %0d/%0d words match Sail)", sigwords, sigwords);
            else $display("FAIL signature %0d/%0d words mismatch", sigmism, sigwords);
        end
        else if (tohost_hit) begin
            if (tohost_val == 1) $display("PASS"); else $display("FAIL tohost=%0d (test %0d)", tohost_val, tohost_val >> 1);
        end
        else if (!halted) begin $display("FAIL timeout"); $finish; end
        else if (expect_cause >= 0) begin
            if (trap_cause == expect_cause) $display("PASS"); else $display("FAIL expected cause %0d got %0d", expect_cause, trap_cause);
        end
        else if (trap_cause == 4'd11 && dut.registers[10] == 0 && dut.registers[17] == 93)
            $display("PASS");
        else $display("FAIL cause=%0d pc=%08x gp(test#)=%0d a0=%0d", trap_cause, trap_pc, dut.registers[3], dut.registers[10]);
        $finish;
    end
endmodule
